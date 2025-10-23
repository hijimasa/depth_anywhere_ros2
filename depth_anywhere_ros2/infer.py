#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import torch
from torchvision import transforms
from numba import njit
import math

# Depth Anywhere のモデル実装パス
from depth_anywhere_ros2.baseline_models.UniFuse.networks import UniFuse
from depth_anywhere_ros2.baseline_models.BiFuseV2 import BiFuse
from depth_anywhere_ros2.baseline_models.HoHoNet.lib.model.hohonet import HoHoNet
from depth_anywhere_ros2.baseline_models.EGformer.models.egformer import EGDepthModel

# equirect→cube 変換ユーティリティ
from depth_anywhere_ros2.utils.Projection import py360_E2C

np.bool = np.bool_
np.float = np.float32
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# JITコンパイルでフィルタ処理を高速化
@njit(fastmath=True)
def apply_mask(pts, rgb):
    N = pts.shape[0]
    out = []
    for i in range(N):
        # if pts[i, 1] <= 1.6 and math.hypot(pts[i, 0], pts[i, 2]) >= 0.1:
        out.append((pts[i, 0], pts[i, 2], pts[i, 1], rgb[i]))
    return out

def load_model(device: str, model_name: str, equi_h: int = 512, equi_w: int = 1024, num_layers: int = 18):
    """Depth Anywhere の各モデルをロードして eval モードに"""
    tkg_robot_launcher_share_dir = get_package_share_directory("depth_anywhere_ros2")
    ckpt_path = os.path.join(tkg_robot_launcher_share_dir, "ckpt")
    model_name = model_name.upper()
    if model_name == 'UNIFUSE':
        net = UniFuse(num_layers=num_layers,
                      equi_h=equi_h, equi_w=equi_w,
                      pretrained=True,
                      max_depth=10.0,
                      fusion_type='cee',
                      se_in_fusion=True)
        ckpt_path = os.path.join(ckpt_path, 'UniFuse_SpatialAudioGen.pth')
    elif model_name == 'BIFUSEV2':
        net = BiFuse.SupervisedCombinedModel('outputs', {'layers':34, 'CE_equi_h':[8,16,32,64,128,256,512]})
        ckpt_path = os.path.join(ckpt_path, 'BiFuseV2_SpatialAudioGen.pth')
    elif model_name == 'HOHONET':
        net = HoHoNet(emb_dim=256,
                      backbone_config={'module':'Resnet','kwargs':{'backbone':'resnet50'}},
                      decode_config={'module':'EfficientHeightReduction'},
                      refine_config={'module':'TransEn','kwargs':{'position_encode':256,'num_layers':1}},
                      modalities_config={'DepthEstimator':{'basis':'dct','n_components':64,'loss':'l1'}})
        net.forward = net.infer
        ckpt_path = os.path.join(ckpt_path, 'HoHoNet.pth')
    elif model_name == 'EGFORMER':
        net = EGDepthModel(hybrid=False)
        ckpt_path = os.path.join(ckpt_path, 'EGFormer.pth')
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    net.to(device)

    # EGFormerの場合、ref_pointもデバイスに移動
    if model_name == 'EGFORMER':
        if hasattr(net, 'ref_point256x512'):
            net.ref_point256x512 = net.ref_point256x512.to(device)
        if hasattr(net, 'ref_point128x256'):
            net.ref_point128x256 = net.ref_point128x256.to(device)
        if hasattr(net, 'ref_point64x128'):
            net.ref_point64x128 = net.ref_point64x128.to(device)
        if hasattr(net, 'ref_point32x64'):
            net.ref_point32x64 = net.ref_point32x64.to(device)
        if hasattr(net, 'ref_point16x32'):
            net.ref_point16x32 = net.ref_point16x32.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt)
    net.eval()
    return net

class DepthAnywherePCL(Node):
    def __init__(self):
        super().__init__('depth_anywhere_pcl')

        # --- ROS パラメータ ---
        self.declare_parameter('model_name',       'UniFuse')  # UniFuseをデフォルトに
        self.declare_parameter('equi_h',           512)
        self.declare_parameter('equi_w',           1024)
        self.declare_parameter('input_h',          256)  # 入力画像の高さ（モデル解像度より小さく）
        self.declare_parameter('input_w',          512)  # 入力画像の幅（モデル解像度より小さく）
        self.declare_parameter('device',           'cuda')
        self.declare_parameter('scale_factor',       2.0)
        self.declare_parameter('use_fp16',         True)
        self.declare_parameter('pcl_downsample',   4)  # 点群を1/4に削減
        self.declare_parameter('num_layers',       18)  # ResNetバックボーン: 18(軽量) or 34(デフォルト)
        self.declare_parameter('frame_skip',       1)  # フレームスキップ（1なら全フレーム処理、2なら2フレームに1回処理）
        
        # 平滑化パラメータ
        self.declare_parameter('spatial_smooth_kernel', 0)  # 空間的平滑化カーネルサイズ（0=無効, 3,5,7など奇数推奨）
        self.declare_parameter('spatial_smooth_method', 'gaussian')  # 空間平滑化手法: 'gaussian', 'bilateral', 'plane_aware'
        self.declare_parameter('bilateral_sigma_color', 0.1)  # bilateral filter色空間sigma（深度差の許容範囲）
        self.declare_parameter('bilateral_sigma_space', 5.0)  # bilateral filter空間sigma（距離の影響範囲）
        self.declare_parameter('plane_threshold', 0.02)  # 平面検出の深度差閾値（メートル）
        self.declare_parameter('temporal_smooth_alpha', 0.0)  # 時間的平滑化係数（0.0=無効, 0.0-1.0: 大きいほど過去フレームの影響大）
        self.declare_parameter('temporal_smooth_frames', 5)  # 時間的平滑化で保持する過去フレーム数

        self.model_name   = self.get_parameter('model_name').get_parameter_value().string_value
        H            = self.get_parameter('equi_h').get_parameter_value().integer_value
        W            = self.get_parameter('equi_w').get_parameter_value().integer_value
        self.input_h = self.get_parameter('input_h').get_parameter_value().integer_value
        self.input_w = self.get_parameter('input_w').get_parameter_value().integer_value
        device       = self.get_parameter('device').get_parameter_value().string_value
        self.scale_factor = self.get_parameter('scale_factor').get_parameter_value().double_value
        self.use_fp16 = self.get_parameter('use_fp16').get_parameter_value().bool_value
        self.pcl_downsample = self.get_parameter('pcl_downsample').get_parameter_value().integer_value
        num_layers = self.get_parameter('num_layers').get_parameter_value().integer_value
        self.frame_skip = self.get_parameter('frame_skip').get_parameter_value().integer_value
        
        # 平滑化パラメータの取得
        self.spatial_kernel = self.get_parameter('spatial_smooth_kernel').get_parameter_value().integer_value
        self.spatial_method = self.get_parameter('spatial_smooth_method').get_parameter_value().string_value
        self.bilateral_sigma_color = self.get_parameter('bilateral_sigma_color').get_parameter_value().double_value
        self.bilateral_sigma_space = self.get_parameter('bilateral_sigma_space').get_parameter_value().double_value
        self.plane_threshold = self.get_parameter('plane_threshold').get_parameter_value().double_value
        self.temporal_alpha = self.get_parameter('temporal_smooth_alpha').get_parameter_value().double_value
        self.temporal_frames = self.get_parameter('temporal_smooth_frames').get_parameter_value().integer_value

        self.frame_count = 0
        
        # 時間的平滑化用のバッファ（過去の深度マップを保持）
        self.depth_history = []
        if self.temporal_alpha > 0.0 and self.temporal_frames > 0:
            self.get_logger().info(f'Temporal smoothing enabled: alpha={self.temporal_alpha}, frames={self.temporal_frames}')
        if self.spatial_kernel > 0:
            self.get_logger().info(f'Spatial smoothing enabled: method={self.spatial_method}, kernel_size={self.spatial_kernel}')

        # モデルロード
        # デバイス設定
        # 'cuda' / 'cpu' の文字列から torch.device を生成
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # モデルロード（self.device と解像度を渡す）
        self.net = load_model(self.device, self.model_name, H, W, num_layers)

        # FP16最適化
        if self.use_fp16 and self.device.type == 'cuda':
            self.net = self.net.half()
            self.get_logger().info('Using FP16 precision for inference')

        self.requires_cube = (self.model_name.upper() == 'UNIFUSE')
        self.get_logger().info(f'Loaded {self.model_name} on {device} (cube={self.requires_cube}, fp16={self.use_fp16})')

        # CUDAストリームの作成
        if self.device.type == 'cuda':
            self.cuda_stream = torch.cuda.Stream()
        else:
            self.cuda_stream = None

        # CvBridge & subscriber/publisher
        self.br  = CvBridge()
        self.sub = self.create_subscription(Image, "image", self.cb_image, 1)
        self.pub = self.create_publisher(PointCloud2, "points", 1)

        # equirectangular→方向ベクトルマップを事前生成（入力解像度ベース）
        self.H = H; self.W = W
        # 点群ダウンサンプリングを考慮した方向ベクトル（入力解像度ベース）
        if self.pcl_downsample > 1:
            # ダウンサンプリング後のサイズで生成
            h_ds = self.input_h // self.pcl_downsample
            w_ds = self.input_w // self.pcl_downsample
            u = (np.arange(w_ds) * self.pcl_downsample + 0.5) / self.input_w * 2 * np.pi - np.pi
            v = np.pi/2 - (np.arange(h_ds) * self.pcl_downsample + 0.5) / self.input_h * np.pi
        else:
            u = (np.arange(self.input_w) + 0.5) / self.input_w * 2 * np.pi - np.pi
            v = np.pi/2 - (np.arange(self.input_h) + 0.5) / self.input_h * np.pi
        uu, vv = np.meshgrid(u, v)
        x = np.cos(vv) * np.sin(uu)
        y = np.sin(vv)
        z = np.cos(vv) * np.cos(uu)
        self.dirs = np.stack((x, y, z), axis=2)  # input_h×input_w×3 or (input_h/ds)×(input_w/ds)×3

        self.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.INT32,   count=1),
        ]

        self.to_tensor  = transforms.ToTensor()
        self.normalize  = transforms.Normalize(mean=MEAN, std=STD)
        if self.requires_cube:
            self.E2C = py360_E2C(equ_h=self.H, equ_w=self.W, face_w=self.H//2)

        self.get_logger().info('Node initialized, waiting for images...')

    def apply_spatial_smoothing(self, depth_map):
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
    
    def apply_temporal_smoothing(self, depth_map):
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

    def cb_image(self, msg: Image):
        # フレームスキップ
        self.frame_count += 1
        if self.frame_skip > 1:
            if self.frame_count % self.frame_skip != 0:
                return
            else:
                self.frame_count = 0

        # 1) 画像取出し → 前処理（入力解像度で処理）
        img_rgb = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # 入力解像度にリサイズ（モデル解像度より小さい場合がある）
        img_rgb = cv2.resize(img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # モデル解像度にアップスケール（必要な場合）
        if self.input_h != self.H or self.input_w != self.W:
            img_for_model = cv2.resize(img_rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        else:
            img_for_model = img_rgb

        rgb_t = self.normalize(self.to_tensor(img_for_model)).unsqueeze(0).to(self.device, non_blocking=True)

        # FP16変換
        if self.use_fp16 and self.device.type == 'cuda':
            rgb_t = rgb_t.half()

        # 2) Cube変換（UniFuseの場合）
        if self.requires_cube:
            cube = self.E2C.run(img_for_model)
            cube_t = self.normalize(self.to_tensor(cube)).unsqueeze(0).to(self.device, non_blocking=True)
            if self.use_fp16 and self.device.type == 'cuda':
                cube_t = cube_t.half()

        # 3) 推論 (torch.ampで自動混合精度)
        with torch.no_grad():
            if self.use_fp16 and self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    if self.requires_cube:
                        out = self.net(rgb_t, cube_t)
                    else:
                        out = self.net(rgb_t)
            else:
                if self.requires_cube:
                    out = self.net(rgb_t, cube_t)
                else:
                    out = self.net(rgb_t)
        if self.model_name.upper() == 'HOHONET':
            depth = out['depth'].squeeze().cpu().numpy()
        else:
            depth = out['pred_depth'].squeeze().cpu().numpy()

        # FP16の場合はFP32に変換（OpenCV互換性のため）
        if depth.dtype == np.float16:
            depth = depth.astype(np.float32)

        # 3) バックプロジェクト → 点群＋色（入力解像度ベース）
        # 入力解像度の画像を使用して点群を生成
        if self.input_h != self.H or self.input_w != self.W:
            # 深度マップを入力解像度にダウンサンプリング
            depth_resized = cv2.resize(depth, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        else:
            depth_resized = depth

        # ダウンサンプリング
        if self.pcl_downsample > 1:
            depth_ds = depth_resized[::self.pcl_downsample, ::self.pcl_downsample]
            img_rgb_ds = img_rgb[::self.pcl_downsample, ::self.pcl_downsample]
        else:
            depth_ds = depth_resized
            img_rgb_ds = img_rgb

        # === 平滑化処理 ===
        # 1. 空間的平滑化（ガウシアンブラー）
        depth_ds = self.apply_spatial_smoothing(depth_ds)

        # 2. 時間的平滑化（過去フレームとの加重平均）
        depth_ds = self.apply_temporal_smoothing(depth_ds)

        if self.model_name.upper() == 'UNIFUSE' or  self.model_name.upper() == 'BIFUSEV2':
            pts    = (self.dirs / (depth_ds[...,None] + 1e-6) * self.scale_factor).reshape(-1, 3)
        else:
            pts    = (self.dirs / (depth_ds[...,None] - depth_ds.min() + 1e-6) * self.scale_factor).reshape(-1, 3)

        colors_img = img_rgb_ds.reshape(-1, 3)
        colors = (colors_img[:,0].astype(np.int32) << 16) | \
                 (colors_img[:,1].astype(np.int32) << 8)  | \
                  colors_img[:,2].astype(np.int32)

       # --- 3) フィルタ & 構造化データ作成 ---
        # JIT 関数でマスク適用 & 結合
        filtered = apply_mask(pts, colors)

        header = msg.header
        header.frame_id = 'camera_link'
        header.stamp = self.get_clock().now().to_msg()
        cloud = pc2.create_cloud(header, self.fields, filtered)

        # 5) パブリッシュ
        self.pub.publish(cloud)

def main(args=None):
    rclpy.init(args=args)
    node = DepthAnywherePCL()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
