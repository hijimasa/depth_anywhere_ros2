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
        if pts[i, 1] <= 1.6 and math.hypot(pts[i, 0], pts[i, 2]) >= 0.1:
            out.append((pts[i, 0], pts[i, 2], pts[i, 1], rgb[i]))
    return out

def load_model(device: str, model_name: str):
    """Depth Anywhere の各モデルをロードして eval モードに"""
    tkg_robot_launcher_share_dir = get_package_share_directory("depth_anywhere_ros2")
    ckpt_path = os.path.join(tkg_robot_launcher_share_dir, "ckpt")
    model_name = model_name.upper()
    if model_name == 'UNIFUSE':
        net = UniFuse(num_layers=18,
                      equi_h=512, equi_w=1024,
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
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt)
    net.eval()
    return net

class DepthAnywherePCL(Node):
    def __init__(self):
        super().__init__('depth_anywhere_pcl')

        # --- ROS パラメータ ---
        self.declare_parameter('model_name',       'UniFuse')
        self.declare_parameter('equi_h',           512)
        self.declare_parameter('equi_w',           1024)
        self.declare_parameter('device',           'cuda')
        self.declare_parameter('scale_factor',       2.0)

        self.model_name   = self.get_parameter('model_name').get_parameter_value().string_value
        H            = self.get_parameter('equi_h').get_parameter_value().integer_value
        W            = self.get_parameter('equi_w').get_parameter_value().integer_value
        device       = self.get_parameter('device').get_parameter_value().string_value
        self.scale_factor = self.get_parameter('scale_factor').get_parameter_value().double_value

        # モデルロード
        # デバイス設定
        # 'cuda' / 'cpu' の文字列から torch.device を生成
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # モデルロード（self.device を渡す）
        self.net = load_model(self.device, self.model_name)
        self.requires_cube = (self.model_name.upper() == 'UNIFUSE')
        self.get_logger().info(f'Loaded {self.model_name} on {device} (cube={self.requires_cube})')

        # CvBridge & subscriber/publisher
        self.br  = CvBridge()
        self.sub = self.create_subscription(Image, "image", self.cb_image, 1)
        self.pub = self.create_publisher(PointCloud2, "points", 1)

        # equirectangular→方向ベクトルマップを事前生成
        self.H = H; self.W = W
        u = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
        v = np.pi/2 - (np.arange(H) + 0.5) / H * np.pi
        uu, vv = np.meshgrid(u, v)
        x = np.cos(vv) * np.sin(uu)
        y = np.sin(vv)
        z = np.cos(vv) * np.cos(uu)
        self.dirs = np.stack((x, y, z), axis=2)  # H×W×3

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

    def cb_image(self, msg: Image):
        # 1) 画像取出し → 前処理
        img_rgb = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img_rgb = cv2.resize(img_rgb, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        rgb_t = self.normalize(self.to_tensor(img_rgb)).unsqueeze(0).to(self.device)   # 1×3×H×W

        # 2) 推論
        with torch.no_grad():
            if self.requires_cube:
                cube = self.E2C.run(img_rgb)
                cube_t = self.normalize(self.to_tensor(cube)).unsqueeze(0).to(self.device)  # 1×3×6H×6H faces
                out = self.net(rgb_t, cube_t)
            else:
                out = self.net(rgb_t)
        if self.model_name.upper() == 'HOHONET':
            depth = out['depth'].squeeze().cpu().numpy()
        else:
            depth = out['pred_depth'].squeeze().cpu().numpy()

        # 3) バックプロジェクト → 点群＋色
        if self.model_name.upper() == 'UNIFUSE' or  self.model_name.upper() == 'BIFUSEV2':
            pts    = (self.dirs / (depth[...,None] + 1e-6) * self.scale_factor).reshape(-1, 3)
        else:
            pts    = (self.dirs / (depth[...,None] - depth.min() + 1e-6) * self.scale_factor).reshape(-1, 3)
            #pts    = (self.dirs * (10.0 - depth[...,None]) / 6.0 * self.scale_factor).reshape(-1, 3)
        #pts    = (self.dirs / (depth[...,None] + 1e-6)).reshape(-1, 3)
        colors = img_rgb.reshape(-1, 3)
        colors = (colors[:,0].astype(np.int32) << 16) | \
                 (colors[:,1].astype(np.int32) << 8)  | \
                  colors[:,2].astype(np.int32)

       # --- 3) フィルタ & 構造化データ作成 ---
        # JIT 関数でマスク適用 & 結合
        filtered = apply_mask(pts, colors)

        header = msg.header
        header.frame_id = 'camera_link'
        header.stamp = self.get_clock().now().to_msg()
        cloud = pc2.create_cloud(header, self.fields, filtered)

        # 5) パブリッシュ
        self.pub.publish(cloud)
        #self.get_logger().debug(f'Published {pts.shape[0]} points')

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
