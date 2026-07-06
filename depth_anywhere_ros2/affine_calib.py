"""床基準のオンライン・アフィン較正。

UniFuse (Depth Anywhere ckpt) の出力はスケール・シフト不定の視差
pred ≈ α·(1/r) + β で、α・β は学習損失がアフィン不変なためフレームや
シーンで揺れる。カメラの床上高さ cam_z が既知なら、俯角 θ の画素が床を
見ているとき真距離 r = cam_z/sinθ が分かるので、床帯の画素から
(x, y) = (1/r, pred) の直線フィットで α・β を毎フレーム推定できる。

tkg_tps_viewer_gl の estimate_depth_affine の移植 + 飽和域除外
（出力は 10·sigmoid なので 0/10 付近はアフィン関係が崩れる）。

前提: カメラは直立（ロール・ピッチ0、ヨーのみ）。俯角は画像の行のみで
決まり、自機方向の除外は列のみで決まる。
"""
import math

import numpy as np


class DepthAffineCalibrator:
    """1カメラ分の α・β を推定・保持する。

    使い方: 毎フレーム update(pred) を呼ぶ。established が True になったら
    alpha/beta を距離変換 r = alpha/(pred - beta) に使う。
    """

    NUM_BINS = 8
    MIN_BIN_SAMPLES = 30
    MIN_BINS_FOR_FIT = 5
    MIN_FLOOR_INLIERS = 200
    FLOOR_Z_TOL = 0.12  # 床インライア判定の |z| 閾値 [m]

    def __init__(self, cam_x, cam_y, cam_z, yaw_deg,
                 min_depression_deg=20.0, max_depression_deg=60.0,
                 smoothing=0.1, exclude_halfwidth_deg=50.0,
                 pred_min=0.3, pred_max=9.7):
        self.cam_x = float(cam_x)
        self.cam_y = float(cam_y)
        self.cam_z = float(cam_z)
        self.yaw = math.radians(float(yaw_deg))
        self.dep_min = math.radians(float(min_depression_deg))
        self.dep_max = math.radians(float(max_depression_deg))
        self.smoothing = float(smoothing)
        self.exclude_cos = math.cos(math.radians(float(exclude_halfwidth_deg)))
        self.pred_min = float(pred_min)
        self.pred_max = float(pred_max)

        self.alpha = None
        self.beta = None
        self._shape = None

    @property
    def established(self):
        return self.alpha is not None

    def _prepare(self, h, w):
        """解像度依存の格子を前計算する。"""
        # 行 → 俯角（カメラ直立なら行だけで決まる）
        lat = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
        sz = -np.sin(lat)  # sin(俯角)。下半分で正
        dep = np.arcsin(np.clip(sz, -1.0, 1.0))
        rows = np.nonzero((dep >= self.dep_min) & (dep <= self.dep_max))[0]
        self._rows = rows
        self._sz = sz[rows].astype(np.float64)                    # (R,)
        self._x = self._sz / self.cam_z                           # (R,) = 1/床距離
        self._bin = np.clip(((dep[rows] - self.dep_min)
                             / (self.dep_max - self.dep_min)
                             * self.NUM_BINS).astype(int),
                            0, self.NUM_BINS - 1)                 # (R,)

        # 列 → 自機（ロボット中心）方向 ±exclude_halfwidth の除外。
        # ビューワと同じ式: 画素方向の水平成分をヨーで回転し to_center と内積
        lon = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        hx = cy * np.sin(lon) - sy * np.cos(lon)
        hy = sy * np.sin(lon) + cy * np.cos(lon)
        tc_norm = math.hypot(self.cam_x, self.cam_y)
        if tc_norm > 1e-6:
            dot = (hx * (-self.cam_x) + hy * (-self.cam_y)) / tc_norm
            self._col_keep = dot <= self.exclude_cos              # (W,)
        else:
            self._col_keep = np.ones(w, dtype=bool)

    @staticmethod
    def _fit(x, y):
        """最小二乗直線 y = a·x + b。特異なら None。"""
        n = x.size
        sx = x.sum()
        sy = y.sum()
        sxx = (x * x).sum()
        sxy = (x * y).sum()
        det = n * sxx - sx * sx
        if abs(det) < 1e-9:
            return None
        a = (n * sxy - sx * sy) / det
        b = (sy * sxx - sx * sxy) / det
        return a, b

    def _robust_floor_line(self, xs, ys):
        """床線の頑健推定（下側包絡線フィット）。

        床はどの方向でも「最遠面」なので pred は最小、つまり真の床線より
        有意に下に来るビンは存在しない。一方、台座・家具は床と同じ切片 β を
        共有し傾きだけ大きい直線に乗るため、汚染ビンが多数派になると
        通常の最小二乗＋対称トリムは台座線にロックする。
        そこで全ペアの候補線のうち「下側外れ点のない」線だけを許容し、
        その中でインライア最多の線を選ぶ（汚染ビンは上側に外れて無視される）。
        """
        n = xs.size
        tol = max(0.05 * float(np.median(ys)), 0.02)
        best_score = None
        best_inl = None
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[j] - xs[i]
                if abs(dx) < 1e-9:
                    continue
                a = (ys[j] - ys[i]) / dx
                if not (a > 0.01):
                    continue
                b = ys[i] - a * xs[i]
                res = ys - (a * xs + b)
                if (res < -3.0 * tol).any():
                    continue  # 床より遠い面は存在しない → 下側外れは非床線
                inl = np.abs(res) <= tol
                if inl.sum() < 3:
                    continue
                score = (int(inl.sum()), -float(np.abs(res[inl]).sum()))
                if best_score is None or score > best_score:
                    best_score = score
                    best_inl = inl
        if best_inl is not None:
            fit = self._fit(xs[best_inl], ys[best_inl])
            if fit is not None and fit[0] > 0.01:
                return fit
        # フォールバック: 全点LS + 対称残差トリム（従来動作）
        fit = self._fit(xs, ys)
        if fit is None:
            return None
        a, b = fit
        res = np.abs(ys - (a * xs + b))
        keep = res <= max(2.0 * np.median(res), 1e-4)
        if keep.sum() >= self.MIN_BINS_FOR_FIT:
            refit = self._fit(xs[keep], ys[keep])
            if refit is not None:
                return refit
        return a, b

    def update(self, pred):
        """床帯から (a, b) を推定して EMA 更新する。

        pred: (H, W) float。モデル出力そのもの（視差ドメイン）。
        返り値: 今回のフレーム単体の (a, b)。フィット不能なら None
        （その場合も確立済みの alpha/beta は保持される）。
        """
        if self.cam_z < 0.1:
            return None
        h, w = pred.shape
        if self._shape != (h, w):
            self._prepare(h, w)
            self._shape = (h, w)
        if self._rows.size == 0:
            return None

        band = pred[self._rows][:, self._col_keep].astype(np.float64)  # (R, C)
        valid = np.isfinite(band) & (band > self.pred_min) & (band < self.pred_max)

        # ビン代表値: 床はその方向の最遠面 = pred の小さい側クラスタなので、
        # 下位5%（スメア等の外れ値）を除いた15パーセンタイルを採る。
        # 台座・家具・自機の残りは近い側 = 大きい側に外れる。
        xs, ys = [], []
        for b in range(self.NUM_BINS):
            rsel = self._bin == b
            vals = band[rsel][valid[rsel]]
            if vals.size < self.MIN_BIN_SAMPLES:
                continue
            vals = np.sort(vals)
            n0 = int(vals.size * 0.05)
            k = min(n0 + int((vals.size - n0) * 0.15), vals.size - 1)
            ys.append(vals[k])
            cnt = valid[rsel].sum(axis=1)
            xs.append((self._x[rsel] * cnt).sum() / cnt.sum())
        if len(xs) < self.MIN_BINS_FOR_FIT:
            return None
        xs = np.array(xs)
        ys = np.array(ys)

        fit = self._robust_floor_line(xs, ys)
        if fit is None:
            return None
        a, b = fit
        if not (a > 0.01) or not math.isfinite(b):
            return None

        # 床インライアによる反復精緻化: 現在の (a, b) で復元した z が床近傍の
        # 画素だけで x = 1/床距離, y = pred を再フィット
        v_all = band[valid]
        x_all = np.broadcast_to(self._x[:, None], band.shape)[valid]
        sz_all = np.broadcast_to(self._sz[:, None], band.shape)[valid]
        for _ in range(2):
            inv = (v_all - b) / a
            ok = inv >= 0.05  # d > 20m は捨てる
            d = np.empty_like(inv)
            d[ok] = 1.0 / inv[ok]
            z = self.cam_z - d * sz_all
            floor = ok & (np.abs(z) < self.FLOOR_Z_TOL)
            if floor.sum() < self.MIN_FLOOR_INLIERS:
                break
            fit = self._fit(x_all[floor], v_all[floor])
            if fit is None or not (fit[0] > 0.01):
                break
            a, b = fit

        if not (a > 0.01) or not math.isfinite(b):
            return None

        if self.alpha is None:
            self.alpha = a
            self.beta = b
        else:
            self.alpha += (a - self.alpha) * self.smoothing
            self.beta += (b - self.beta) * self.smoothing
        return a, b
