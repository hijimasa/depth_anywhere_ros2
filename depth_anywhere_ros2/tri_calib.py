"""2カメラ共通視野の三角測量によるオンライン深度較正。

UniFuse (Depth Anywhere ckpt) の出力 pred はスケール・シフト不定の視差
（pred ≈ a·(1/r) + b、a,b はシーン・フレームで揺れる）。2カメラの位置関係
（基線 ≈ 0.96m）は実測で既知なので、両カメラに共通に写る物体を特徴マッチング
→三角測量すると真距離 r が得られ、(pred, 1/r) の直線フィットで a,b を
オンライン推定できる。床の見え方（台座・自機による汚染）に依存しないのが利点。

- 対応点は基線に垂直な2セクタ（視差最大・自機遮蔽なし。既定配置なら右前と左後ろ）
  からのみ採る。基線方向に近い点は視差が退化するため使わない。
- 画素→方向の変換は tps_viewer_gl で数値検証済みの「設定yaw + 90°」規約
  （tps_viewer_gl_node.cpp の cam*_rotation_ と同じ）。
- rosbag 実データで検証済み: スキュー門(距離5%+3cm)を通る対応点が
  フレームペアあたり数十点得られ、床基準の較正値と同オーダーで一致。

処理は重い（ORB+マッチングで数十ms）ため、ノードからは submit() で
低レートに投げ、別スレッドで処理する。結果は alpha/beta に反映される。
"""
import math
import threading
from collections import deque

import numpy as np
import cv2


class TriangulationCalibrator:
    """2カメラ専用。alpha/beta は r = alpha/(pred - beta) にそのまま使う。"""

    PRED_MIN = 0.3   # 10·sigmoid の飽和域はアフィン関係が崩れるため除外
    PRED_MAX = 9.7
    SECTOR_HALF_DEG = 40.0
    SECTOR_LAT_HALF_DEG = 45.0
    RATIO_TEST = 0.75
    MAX_HAMMING = 60
    R_MIN = 0.5      # 三角測量距離の採用範囲 [m]
    R_MAX = 9.0
    MIN_X_SPREAD = 0.12   # 1/r の広がりがこれ未満なら切片が定まらないので棄却

    FIT_WINDOW = 7        # tickフィットの中央値ウィンドウ（単発の外れフィットを除去）
    GAIN_FULL_SAMPLES = 30  # このtick採用点数でゲイン最大（少ないtickは弱く反映）

    def __init__(self, positions, yaws_deg,
                 smoothing=0.2, min_samples=60, buffer_size=500,
                 nfeatures=1500):
        """positions: [(x,y,z)]*2 ロボット座標系 [m]。yaws_deg: yaml と同じ値。"""
        self.pos = [np.asarray(p, dtype=np.float64) for p in positions]
        # ビューワ検証済み規約: 実効ヨー = 設定yaw + 90°
        self.yaw = [math.radians(y + 90.0) for y in yaws_deg]
        self.smoothing = float(smoothing)
        self.min_samples = int(min_samples)

        base = self.pos[1][:2] - self.pos[0][:2]
        az_base = math.atan2(base[1], base[0])
        # 基線に垂直な2方位（ロボット座標系）
        self.sector_az = (az_base + math.pi / 2, az_base - math.pi / 2)

        self.orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=8)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.alpha = [None, None]
        self.beta = [None, None]
        self.samples = [deque(maxlen=buffer_size), deque(maxlen=buffer_size)]
        # tickごとの生フィット履歴（中央値でtick単位の外れを除去）
        self._fits = [deque(maxlen=self.FIT_WINDOW), deque(maxlen=self.FIT_WINDOW)]
        self.stats = {'ticks': 0, 'matches': 0, 'accepted': 0}

        self._masks = {}   # (cam, h, w) -> セクタマスク
        self._lock = threading.Lock()
        self._busy = False

    def established(self, cam):
        return self.alpha[cam] is not None

    # ---- 幾何 ----

    def _dirs(self, uv, h, w, cam):
        """画素 (N,2 col,row) → ロボット座標系方向 (N,3)。ビューワ規約"""
        lat = np.pi / 2 - (uv[:, 1] + 0.5) / h * np.pi
        lon = (uv[:, 0] + 0.5) / w * 2 * np.pi - np.pi
        dx = np.cos(lat) * np.sin(lon)
        dy = np.cos(lat) * np.cos(lon)
        dz = np.sin(lat)
        cy, sy = math.cos(self.yaw[cam]), math.sin(self.yaw[cam])
        return np.stack([cy * dx - sy * dy, sy * dx + cy * dy, dz], axis=1)

    def _sector_masks(self, cam, h, w):
        """セクタごとのマスクのリスト。セクタ別にマッチングすることで
        反対側セクタとの誤対応が ratio test を汚すのを防ぐ"""
        key = (cam, h, w)
        if key not in self._masks:
            rows, cols = np.mgrid[0:h, 0:w]
            uv = np.stack([cols.ravel(), rows.ravel()], axis=1)
            d = self._dirs(uv, h, w, cam)
            az = np.arctan2(d[:, 1], d[:, 0])
            lat = np.arcsin(np.clip(d[:, 2], -1, 1))
            lat_ok = np.abs(lat) < math.radians(self.SECTOR_LAT_HALF_DEG)
            masks = []
            for center in self.sector_az:
                diff = np.abs((az - center + np.pi) % (2 * np.pi) - np.pi)
                m = (diff < math.radians(self.SECTOR_HALF_DEG)) & lat_ok
                masks.append(m.reshape(h, w).astype(np.uint8) * 255)
            self._masks[key] = masks
        return self._masks[key]

    # ---- 実行 ----

    def submit(self, gray0, gray1, pred0, pred1):
        """較正1回分を別スレッドで実行（前回が処理中ならスキップ）。

        gray*: 各カメラのグレースケール equirect（解像度は pred と違ってよい）
        pred*: モデル出力（視差ドメイン、入力解像度）
        """
        if self._busy:
            return False
        self._busy = True
        t = threading.Thread(
            target=self._work,
            args=(gray0.copy(), gray1.copy(), pred0.copy(), pred1.copy()),
            daemon=True)
        t.start()
        return True

    def _work(self, g0, g1, p0, p1):
        try:
            self._tick(g0, g1, (p0, p1))
        except Exception:
            pass  # 較正はベストエフォート。失敗しても推論は止めない
        finally:
            self._busy = False

    def _tick(self, g0, g1, preds):
        h0, w0 = g0.shape
        h1, w1 = g1.shape
        uv0, uv1 = [], []
        for m0, m1 in zip(self._sector_masks(0, h0, w0),
                          self._sector_masks(1, h1, w1)):
            kp0, des0 = self.orb.detectAndCompute(g0, m0)
            kp1, des1 = self.orb.detectAndCompute(g1, m1)
            if des0 is None or des1 is None or len(des0) < 8 or len(des1) < 8:
                continue
            for pair in self.bf.knnMatch(des0, des1, k=2):
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.MAX_HAMMING and m.distance < self.RATIO_TEST * n.distance:
                    uv0.append(kp0[m.queryIdx].pt)
                    uv1.append(kp1[m.trainIdx].pt)
        if len(uv0) < 8:
            return
        uv0 = np.array(uv0)
        uv1 = np.array(uv1)

        d0 = self._dirs(uv0, h0, w0, 0)
        d1 = self._dirs(uv1, h1, w1, 1)

        # ベクトル化した2レイ最近接（三角測量）
        b = np.sum(d0 * d1, axis=1)
        w_vec = self.pos[0] - self.pos[1]
        dw0 = d0 @ w_vec
        dw1 = d1 @ w_vec
        det = 1.0 - b * b
        t0 = (b * dw1 - dw0) / np.maximum(det, 1e-9)
        t1 = (dw1 - b * dw0) / np.maximum(det, 1e-9)
        q0 = self.pos[0] + t0[:, None] * d0
        q1 = self.pos[1] + t1[:, None] * d1
        skew = np.linalg.norm(q0 - q1, axis=1)
        ok = ((det > 1e-4)
              & (t0 > self.R_MIN) & (t0 < self.R_MAX)
              & (t1 > self.R_MIN) & (t1 < self.R_MAX)
              & (skew < 0.05 * np.minimum(t0, t1) + 0.03))

        accepted = 0
        with self._lock:
            self.stats['ticks'] += 1
            self.stats['matches'] += len(uv0)
            for cam, (uv, hh, ww, r) in enumerate(
                    ((uv0, h0, w0, t0), (uv1, h1, w1, t1))):
                pmap = preds[cam]
                ph, pw = pmap.shape
                dr = (uv[ok, 1] / hh * ph).astype(int).clip(0, ph - 1)
                dc = (uv[ok, 0] / ww * pw).astype(int).clip(0, pw - 1)
                pv = pmap[dr, dc]
                rv = r[ok]
                good = (np.isfinite(pv)
                        & (pv > self.PRED_MIN) & (pv < self.PRED_MAX))
                n_new = int(good.sum())
                accepted += n_new
                for p, rr in zip(pv[good], rv[good]):
                    self.samples[cam].append((float(p), 1.0 / float(rr)))
                self._fit(cam, n_new)
            self.stats['accepted'] += accepted

    def _fit(self, cam, n_new):
        """蓄積サンプルから pred = a·(1/r) + b をロバスト推定し EMA 更新。

        最小二乗はtickごとのサンプル構成変化に敏感で、EMA後でも α が±25%
        揺れた（2026-07-07 bag で実測。r=α/(pred−β) は遠方ほど β に過敏で
        画面の暴れになる）。そのため多段でロバスト化する:
          1) Theil-Sen（ペア傾きの中央値）— 単発の外れサンプルに鈍感
          2) 直近tickフィットの中央値ウィンドウ — 単発の外れtickを丸ごと除去
          3) 採用点数に応じた適応ゲイン — 貧弱なtickは弱くしか反映しない
        """
        if len(self.samples[cam]) < self.min_samples or n_new <= 0:
            return
        arr = np.array(self.samples[cam])
        y, x = arr[:, 0], arr[:, 1]
        if x.max() - x.min() < self.MIN_X_SPREAD:
            return  # 距離の多様性不足。切片が定まらない
        # 固定グリッド（1/r 空間）ビンの中央値に直線を当てる。ビン位置が
        # データによらず固定なので、tickごとのサンプル構成変化に鈍感。
        # 各ビン中央値は外れ値に頑強で、Theil-Sen のような減衰バイアスもない
        edges = np.linspace(1.0 / self.R_MAX, 1.0 / self.R_MIN, 21)
        bi = np.digitize(x, edges) - 1
        xs, ys, ws = [], [], []
        for k in range(len(edges) - 1):
            sel = bi == k
            if sel.sum() < 6:
                continue
            xs.append(float(np.median(x[sel])))
            ys.append(float(np.median(y[sel])))
            ws.append(float(sel.sum()))
        if len(xs) < 3 or max(xs) - min(xs) < self.MIN_X_SPREAD:
            return
        xs = np.array(xs)
        ys = np.array(ys)
        ws = np.sqrt(np.array(ws))  # 点数の平方根で重み付け
        A = np.stack([xs, np.ones_like(xs)], axis=1) * ws[:, None]
        sol, *_ = np.linalg.lstsq(A, ys * ws, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if not (a > 0.05) or not math.isfinite(b):
            return
        self._fits[cam].append((a, b))
        fits = np.array(self._fits[cam])
        a_t = float(np.median(fits[:, 0]))
        b_t = float(np.median(fits[:, 1]))
        if self.alpha[cam] is None:
            # 初期の不安定なフィットで確定しないよう、ウィンドウが溜まってから確立
            if len(self._fits[cam]) >= 3:
                self.alpha[cam] = a_t
                self.beta[cam] = b_t
        else:
            gain = self.smoothing * min(1.0, n_new / self.GAIN_FULL_SAMPLES)
            self.alpha[cam] += (a_t - self.alpha[cam]) * gain
            self.beta[cam] += (b_t - self.beta[cam]) * gain
