"""深度マップの空間・時間平滑化（infer / infer_multi 共用）"""
import cv2
import numpy as np


class DepthSmoother:
    """深度マップに空間的・時間的平滑化を適用する。

    時間平滑化の履歴を内部に持つため、カメラ1台につき1インスタンス使うこと。
    """

    def __init__(self, spatial_kernel=0, spatial_method='gaussian',
                 bilateral_sigma_color=0.1, bilateral_sigma_space=5.0,
                 plane_threshold=0.02,
                 temporal_alpha=0.0, temporal_frames=5):
        self.spatial_kernel = spatial_kernel
        self.spatial_method = spatial_method
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.plane_threshold = plane_threshold
        self.temporal_alpha = temporal_alpha
        self.temporal_frames = temporal_frames
        self.depth_history = []

    def apply(self, depth_map):
        depth_map = self.apply_spatial(depth_map)
        depth_map = self.apply_temporal(depth_map)
        return depth_map

    def apply_spatial(self, depth_map):
        """空間的平滑化: 複数の手法から選択"""
        # 無効値や偶数カーネルは無効化
        if self.spatial_kernel <= 0 or self.spatial_kernel % 2 == 0:
            return depth_map

        method = self.spatial_method.lower()

        if method == 'bilateral':
            return self._apply_bilateral_filter(depth_map)
        elif method == 'plane_aware':
            return self._apply_plane_aware_smoothing(depth_map)
        else:  # 'gaussian' or default
            return self._apply_gaussian_smoothing(depth_map)

    def _apply_gaussian_smoothing(self, depth_map):
        """ガウシアンブラー（NaN対応）"""
        # 有限値マスク
        finite_mask = np.isfinite(depth_map).astype(np.float32)
        depth_zeroed = np.where(finite_mask, depth_map, 0.0).astype(np.float32)

        k = self.spatial_kernel
        depth_blurred = cv2.GaussianBlur(depth_zeroed, (k, k), 0)
        mask_blurred = cv2.GaussianBlur(finite_mask, (k, k), 0)

        eps = 1e-6
        safe_mask = np.where(mask_blurred > eps, mask_blurred, 0.0)

        smoothed = np.zeros_like(depth_blurred, dtype=np.float32)
        valid = safe_mask > 0
        smoothed[valid] = depth_blurred[valid] / safe_mask[valid]
        smoothed[~np.isfinite(smoothed)] = np.nan

        return smoothed

    def _apply_bilateral_filter(self, depth_map):
        """Bilateral Filter（エッジ保存型平滑化、NaN対応）

        床などの平面部分は滑らかにしながら、壁との境界などの深度不連続は保持する。
        """
        # 有限値マスク
        finite_mask = np.isfinite(depth_map).astype(np.float32)
        depth_zeroed = np.where(finite_mask, depth_map, 0.0).astype(np.float32)

        # OpenCVのbilateralFilterはNaNに対応していないため、有限値のみで処理
        # sigma_color: 深度値の差がこの範囲内なら同じ平面とみなす（小さいほどエッジ保持が強い）
        # sigma_space: 空間的な近傍範囲（大きいほど広範囲で平滑化）

        # カーネルサイズをdiameterとして使用
        d = self.spatial_kernel
        sigma_color = self.bilateral_sigma_color * 255.0  # OpenCVは0-255スケールを想定
        sigma_space = self.bilateral_sigma_space

        # 深度値を0-1の範囲に正規化してからbilateralFilterを適用
        depth_min = np.nanmin(depth_zeroed[finite_mask > 0]) if np.any(finite_mask > 0) else 0.0
        depth_max = np.nanmax(depth_zeroed[finite_mask > 0]) if np.any(finite_mask > 0) else 1.0
        depth_range = depth_max - depth_min

        if depth_range > 1e-6:
            depth_normalized = ((depth_zeroed - depth_min) / depth_range * 255.0).astype(np.float32)
        else:
            depth_normalized = depth_zeroed.astype(np.float32)

        # Bilateral filter適用
        depth_filtered = cv2.bilateralFilter(depth_normalized, d, sigma_color, sigma_space)

        # 元のスケールに戻す
        if depth_range > 1e-6:
            depth_filtered = depth_filtered / 255.0 * depth_range + depth_min

        # マスクで正規化（bilateral filterの境界処理を補正）
        mask_blurred = cv2.GaussianBlur(finite_mask, (self.spatial_kernel, self.spatial_kernel), 0)
        eps = 1e-6

        smoothed = np.zeros_like(depth_filtered, dtype=np.float32)
        valid = (mask_blurred > eps) & (finite_mask > 0)
        smoothed[valid] = depth_filtered[valid]

        # 無効ポイントはNaNを保持
        smoothed[~valid] = np.nan

        return smoothed

    def _apply_plane_aware_smoothing(self, depth_map):
        """平面保持型平滑化

        局所的な深度勾配を計算し、平面部分（床など）は強く平滑化、
        エッジ部分（壁との境界）は弱く平滑化する。
        """
        # 有限値マスク
        finite_mask = np.isfinite(depth_map).astype(np.float32)
        depth_zeroed = np.where(finite_mask, depth_map, 0.0).astype(np.float32)

        # 深度勾配を計算（Sobelフィルタ）
        grad_x = cv2.Sobel(depth_zeroed, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_zeroed, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 勾配が小さい部分を平面として検出
        plane_mask = (grad_magnitude < self.plane_threshold).astype(np.float32)

        # 平面部分には強いガウシアン平滑化
        k_strong = max(self.spatial_kernel, 5)  # より大きなカーネル
        depth_strong = cv2.GaussianBlur(depth_zeroed, (k_strong, k_strong), 0)

        # エッジ部分には弱い平滑化
        k_weak = 3
        depth_weak = cv2.GaussianBlur(depth_zeroed, (k_weak, k_weak), 0)

        # 平面マスクで重み付け合成
        # plane_mask が 1 に近いほど strong smoothing を適用
        smoothed = plane_mask * depth_strong + (1.0 - plane_mask) * depth_weak

        # マスクで正規化
        mask_blurred = cv2.GaussianBlur(finite_mask, (self.spatial_kernel, self.spatial_kernel), 0)
        eps = 1e-6
        safe_mask = np.where(mask_blurred > eps, mask_blurred, 0.0)

        result = np.zeros_like(smoothed, dtype=np.float32)
        valid = safe_mask > 0
        result[valid] = smoothed[valid] / safe_mask[valid]

        # 無効ポイントはNaNを保持
        result[~np.isfinite(result)] = np.nan

        return result

    def apply_temporal(self, depth_map):
        """時間的平滑化: 過去フレームとの加重平均"""
        if self.temporal_alpha <= 0.0 or self.temporal_frames <= 0:
            return depth_map

        # 現在のフレームを履歴に追加
        self.depth_history.append(depth_map.copy())

        # 指定フレーム数を超えたら古いものを削除
        if len(self.depth_history) > self.temporal_frames:
            self.depth_history.pop(0)

        # 履歴が1つしかない場合はそのまま返す
        if len(self.depth_history) == 1:
            return depth_map

        # 指数移動平均 (EMA) を計算
        # alpha が大きいほど過去の影響が大きい
        smoothed = self.depth_history[0].copy()
        for i in range(1, len(self.depth_history)):
            smoothed = self.temporal_alpha * smoothed + (1.0 - self.temporal_alpha) * self.depth_history[i]

        return smoothed
