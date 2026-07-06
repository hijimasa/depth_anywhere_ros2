#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import torch
from torchvision import transforms

# Depth Anywhere のモデル実装パス
from depth_anywhere_ros2.baseline_models.UniFuse.networks import UniFuse
from depth_anywhere_ros2.baseline_models.BiFuseV2 import BiFuse
from depth_anywhere_ros2.baseline_models.HoHoNet.lib.model.hohonet import HoHoNet
from depth_anywhere_ros2.baseline_models.EGformer.models.egformer import EGDepthModel

# equirect→cube 変換ユーティリティ
from depth_anywhere_ros2.utils.Projection import py360_E2C
from depth_anywhere_ros2.smoothing import DepthSmoother

np.bool = np.bool_
np.float = np.float32
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

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
    # Cube2Equirec の sample_grid は解像度依存だが __init__ で再計算されるため、
    # 形状が一致するキーだけロードする。これで equi_h/equi_w を
    # 学習時の 512x1024 以外に設定しても既存の pth がそのまま使える。
    model_sd = net.state_dict()
    loadable = {k: v for k, v in ckpt.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    net.load_state_dict(loadable, strict=False)
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
        self.declare_parameter('publish_pointcloud', True)   # 従来の PointCloud2 出力
        self.declare_parameter('publish_depth',      True)   # 半径マップ(32FC1 Image)出力。tkg_tps_viewer_gl の depth モード用
        
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
        self.publish_pointcloud = self.get_parameter('publish_pointcloud').get_parameter_value().bool_value
        self.publish_depth = self.get_parameter('publish_depth').get_parameter_value().bool_value
        
        # 平滑化パラメータの取得
        self.spatial_kernel = self.get_parameter('spatial_smooth_kernel').get_parameter_value().integer_value
        self.spatial_method = self.get_parameter('spatial_smooth_method').get_parameter_value().string_value
        self.bilateral_sigma_color = self.get_parameter('bilateral_sigma_color').get_parameter_value().double_value
        self.bilateral_sigma_space = self.get_parameter('bilateral_sigma_space').get_parameter_value().double_value
        self.plane_threshold = self.get_parameter('plane_threshold').get_parameter_value().double_value
        self.temporal_alpha = self.get_parameter('temporal_smooth_alpha').get_parameter_value().double_value
        self.temporal_frames = self.get_parameter('temporal_smooth_frames').get_parameter_value().integer_value

        self.frame_count = 0

        self.smoother = DepthSmoother(
            spatial_kernel=self.spatial_kernel,
            spatial_method=self.spatial_method,
            bilateral_sigma_color=self.bilateral_sigma_color,
            bilateral_sigma_space=self.bilateral_sigma_space,
            plane_threshold=self.plane_threshold,
            temporal_alpha=self.temporal_alpha,
            temporal_frames=self.temporal_frames,
        )
        if self.temporal_alpha > 0.0 and self.temporal_frames > 0:
            self.get_logger().info(f'Temporal smoothing enabled: alpha={self.temporal_alpha}, frames={self.temporal_frames}')
        if self.spatial_kernel > 0:
            self.get_logger().info(f'Spatial smoothing enabled: method={self.spatial_method}, kernel_size={self.spatial_kernel}')

        # モデルロード
        # デバイス設定
        # 'cuda' / 'cpu' の文字列から torch.device を生成
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            # 入力形状が固定なので autotuner で最速の conv アルゴリズムを選ばせる
            torch.backends.cudnn.benchmark = True
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
        self.pub_depth = self.create_publisher(Image, "depth", 1)

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
        self.point_dtype = np.dtype([('x', np.float32), ('y', np.float32),
                                     ('z', np.float32), ('rgb', np.int32)])

        self.to_tensor  = transforms.ToTensor()
        self.normalize  = transforms.Normalize(mean=MEAN, std=STD)
        if self.requires_cube:
            self.E2C = py360_E2C(equ_h=self.H, equ_w=self.W, face_w=self.H//2)

        self.get_logger().info('Node initialized, waiting for images...')

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

        # 3) 深度マップを入力解像度に揃え、平滑化してから半径マップに変換
        if self.input_h != self.H or self.input_w != self.W:
            depth_resized = cv2.resize(depth, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        else:
            depth_resized = depth

        # === 平滑化処理（全解像度で実施） ===
        depth_resized = self.smoother.apply(depth_resized)

        # pred_depth は「シフト付き逆深度」(近いほど大、遠いほど小。実データで検証済み)。
        # そのまま配信し、絶対距離への変換 d = α/(r−β) はビューワ側の
        # 床基準アフィン較正が行う（β があるため単純な逆数は不可）。
        radius = (self.scale_factor * depth_resized).astype(np.float32)

        header = msg.header
        header.frame_id = 'camera_link'
        header.stamp = self.get_clock().now().to_msg()

        # 4a) 半径マップを 32FC1 Image でパブリッシュ（tkg_tps_viewer_gl の depth モード用）
        if self.publish_depth:
            depth_msg = self.br.cv2_to_imgmsg(radius, encoding='32FC1')
            depth_msg.header = header
            self.pub_depth.publish(depth_msg)

        # 4b) PointCloud2 パブリッシュ（従来の点群コンシューマ用）
        if self.publish_pointcloud:
            if self.pcl_downsample > 1:
                radius_ds = radius[::self.pcl_downsample, ::self.pcl_downsample]
                img_rgb_ds = img_rgb[::self.pcl_downsample, ::self.pcl_downsample]
            else:
                radius_ds = radius
                img_rgb_ds = img_rgb

            pts = (self.dirs * radius_ds[..., None]).reshape(-1, 3)

            colors_img = img_rgb_ds.reshape(-1, 3)
            colors = (colors_img[:, 0].astype(np.int32) << 16) | \
                     (colors_img[:, 1].astype(np.int32) << 8) | \
                     colors_img[:, 2].astype(np.int32)

            # ベクトル化した PointCloud2 生成（点ごとの Python ループを排除）
            n = pts.shape[0]
            cloud_arr = np.empty(n, dtype=self.point_dtype)
            cloud_arr['x'] = pts[:, 0]
            cloud_arr['y'] = pts[:, 2]
            cloud_arr['z'] = pts[:, 1]
            cloud_arr['rgb'] = colors

            cloud = PointCloud2(
                header=header,
                height=1,
                width=n,
                fields=self.fields,
                is_bigendian=False,
                point_step=cloud_arr.itemsize,
                row_step=cloud_arr.itemsize * n,
                is_dense=False,
                data=cloud_arr.tobytes(),
            )
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
