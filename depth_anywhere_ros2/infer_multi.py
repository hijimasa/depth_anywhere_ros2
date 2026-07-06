#!/usr/bin/env python3
"""複数の全方位カメラを1ノードでバッチ推論する深度推定ノード。

- 2台分の画像を batch=2 で一度に推論し、GPUの利用効率を上げる（従来はカメラ毎に
  別ノード・別モデルで、同一GPUを取り合っていた）。
- backend パラメータで PyTorch / TensorRT を切り替えられる。TensorRT エンジンは
  scripts/export_unifuse_onnx.py --batch 2 で作った ONNX を trtexec で変換して用意する。

トピック（remapで実配線する）:
  購読: image_0, image_1, ...
  配信: depth_0, depth_1, ...   (32FC1 半径マップ [m])
        points_0, points_1, ... (publish_pointcloud 時のみ)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge

import cv2
import torch

from depth_anywhere_ros2.infer import load_model, MEAN, STD
from depth_anywhere_ros2.smoothing import DepthSmoother
from depth_anywhere_ros2.utils.Projection import py360_E2C

MEAN_ARR = np.array(MEAN, dtype=np.float32)
STD_ARR = np.array(STD, dtype=np.float32)


def normalize_batch(images):
    """uint8 HWC 画像リスト → 正規化済み (B,3,H,W) float32 配列"""
    batch = np.stack(images).astype(np.float32) / 255.0
    batch = (batch - MEAN_ARR) / STD_ARR
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


class TorchBackend:
    """PyTorch でのバッチ推論"""

    def __init__(self, device, model_name, equi_h, equi_w, num_layers, use_fp16):
        self.device = device
        self.use_fp16 = use_fp16 and device.type == 'cuda'
        self.net = load_model(device, model_name, equi_h, equi_w, num_layers)
        if self.use_fp16:
            self.net = self.net.half()
        self.requires_cube = (model_name.upper() == 'UNIFUSE')
        self.model_name = model_name.upper()

    def infer(self, equi_np, cube_np):
        equi_t = torch.from_numpy(equi_np).to(self.device, non_blocking=True)
        cube_t = torch.from_numpy(cube_np).to(self.device, non_blocking=True) \
            if cube_np is not None else None
        if self.use_fp16:
            equi_t = equi_t.half()
            if cube_t is not None:
                cube_t = cube_t.half()
        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    out = self.net(equi_t, cube_t) if self.requires_cube else self.net(equi_t)
            else:
                out = self.net(equi_t, cube_t) if self.requires_cube else self.net(equi_t)
        key = 'depth' if self.model_name == 'HOHONET' else 'pred_depth'
        depth = out[key].squeeze(1).float().cpu().numpy()  # (B,H,W)
        return depth


class TRTBackend:
    """TensorRT エンジンでのバッチ推論。

    入出力バッファには torch の CUDA テンソルを使う（pycuda 依存を避ける）。
    エンジンは静的形状 (B,3,H,W)/(B,3,H/2,3H) → (B,1,H,W) を想定。
    """

    _TRT_TO_TORCH_DTYPE = None

    def __init__(self, engine_path, batch, equi_h, equi_w, logger):
        import tensorrt as trt
        self.trt = trt
        TRTBackend._TRT_TO_TORCH_DTYPE = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
        }

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'TensorRT エンジンの読み込みに失敗: {engine_path}')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        expected = {
            'equi': (batch, 3, equi_h, equi_w),
            'cube': (batch, 3, equi_h // 2, (equi_h // 2) * 6),
            'depth': (batch, 1, equi_h, equi_w),
        }

        # TRT 8.5+ の tensor 名 API を優先し、古い binding API にフォールバック
        self.use_v3 = hasattr(self.engine, 'num_io_tensors')
        self.buffers = {}
        names = []
        if self.use_v3:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        else:
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]

        for name in names:
            if self.use_v3:
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = self.engine.get_tensor_dtype(name)
            else:
                idx = self.engine.get_binding_index(name)
                shape = tuple(self.engine.get_binding_shape(idx))
                dtype = self.engine.get_binding_dtype(idx)
            torch_dtype = TRTBackend._TRT_TO_TORCH_DTYPE.get(dtype, torch.float32)
            if name in expected and shape != expected[name]:
                raise RuntimeError(
                    f'TensorRT エンジンの {name} 形状 {shape} が期待値 {expected[name]} と不一致。'
                    f'export_unifuse_onnx.py --batch {batch} --height {equi_h} --width {equi_w} '
                    f'で作り直してください')
            self.buffers[name] = torch.empty(shape, dtype=torch_dtype, device='cuda')
            logger.info(f'TRT tensor {name}: shape={shape}, dtype={dtype}')

        if self.use_v3:
            for name, buf in self.buffers.items():
                self.context.set_tensor_address(name, buf.data_ptr())
        else:
            self.bindings = [0] * self.engine.num_bindings
            for name, buf in self.buffers.items():
                self.bindings[self.engine.get_binding_index(name)] = buf.data_ptr()

    def infer(self, equi_np, cube_np):
        with torch.cuda.stream(self.stream):
            self.buffers['equi'].copy_(
                torch.from_numpy(equi_np).to(self.buffers['equi'].dtype), non_blocking=True)
            self.buffers['cube'].copy_(
                torch.from_numpy(cube_np).to(self.buffers['cube'].dtype), non_blocking=True)
            if self.use_v3:
                self.context.execute_async_v3(self.stream.cuda_stream)
            else:
                self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buffers['depth'].squeeze(1).float().cpu().numpy()  # (B,H,W)


class DepthAnywhereMulti(Node):
    def __init__(self):
        super().__init__('depth_anywhere_multi')

        self.declare_parameter('num_cameras',      2)
        self.declare_parameter('model_name',       'UniFuse')
        self.declare_parameter('equi_h',           256)
        self.declare_parameter('equi_w',           512)
        self.declare_parameter('input_h',          256)
        self.declare_parameter('input_w',          512)
        self.declare_parameter('device',           'cuda')
        self.declare_parameter('scale_factor',     2.0)
        self.declare_parameter('use_fp16',         True)
        self.declare_parameter('num_layers',       18)
        self.declare_parameter('frame_skip',       1)
        self.declare_parameter('pcl_downsample',   4)
        self.declare_parameter('publish_pointcloud', False)
        self.declare_parameter('publish_depth',      True)
        # backend: 'pytorch' | 'tensorrt'
        self.declare_parameter('backend',          'pytorch')
        self.declare_parameter('trt_engine_path',  '')
        # 平滑化パラメータ（infer.py と同じ意味、全カメラ共通）
        self.declare_parameter('spatial_smooth_kernel', 0)
        self.declare_parameter('spatial_smooth_method', 'gaussian')
        self.declare_parameter('bilateral_sigma_color', 0.1)
        self.declare_parameter('bilateral_sigma_space', 5.0)
        self.declare_parameter('plane_threshold', 0.02)
        self.declare_parameter('temporal_smooth_alpha', 0.0)
        self.declare_parameter('temporal_smooth_frames', 5)

        gp = lambda n: self.get_parameter(n).value
        self.num_cameras = gp('num_cameras')
        self.model_name = gp('model_name')
        self.H = gp('equi_h')
        self.W = gp('equi_w')
        self.input_h = gp('input_h')
        self.input_w = gp('input_w')
        self.scale_factor = float(gp('scale_factor'))
        self.use_fp16 = gp('use_fp16')
        self.frame_skip = gp('frame_skip')
        self.pcl_downsample = gp('pcl_downsample')
        self.publish_pointcloud = gp('publish_pointcloud')
        self.publish_depth = gp('publish_depth')
        backend_name = gp('backend').lower()

        self.device = torch.device(gp('device') if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        if self.model_name.upper() != 'UNIFUSE':
            raise ValueError('infer_multi は現状 UniFuse のみ対応（cube入力前提のため）')

        # 推論バックエンド
        if backend_name == 'tensorrt':
            engine_path = gp('trt_engine_path')
            self.backend = TRTBackend(engine_path, self.num_cameras, self.H, self.W,
                                      self.get_logger())
            self.get_logger().info(f'TensorRT backend: {engine_path}')
        else:
            self.backend = TorchBackend(self.device, self.model_name, self.H, self.W,
                                        gp('num_layers'), self.use_fp16)
            self.get_logger().info(f'PyTorch backend (fp16={self.use_fp16})')

        # カメラごとの状態
        self.br = CvBridge()
        self.latest_msgs = [None] * self.num_cameras
        self.smoothers = [DepthSmoother(
            spatial_kernel=gp('spatial_smooth_kernel'),
            spatial_method=gp('spatial_smooth_method'),
            bilateral_sigma_color=gp('bilateral_sigma_color'),
            bilateral_sigma_space=gp('bilateral_sigma_space'),
            plane_threshold=gp('plane_threshold'),
            temporal_alpha=gp('temporal_smooth_alpha'),
            temporal_frames=gp('temporal_smooth_frames'),
        ) for _ in range(self.num_cameras)]

        self.subs = []
        self.pubs_depth = []
        self.pubs_points = []
        for i in range(self.num_cameras):
            # カメラ0の到着で処理をトリガ（他カメラは最新値を使う）
            trigger = (i == 0)
            self.subs.append(self.create_subscription(
                Image, f'image_{i}',
                (lambda idx, trig: lambda msg: self.cb_image(idx, msg, trig))(i, trigger), 1))
            self.pubs_depth.append(self.create_publisher(Image, f'depth_{i}', 1))
            self.pubs_points.append(self.create_publisher(PointCloud2, f'points_{i}', 1))

        self.E2C = py360_E2C(equ_h=self.H, equ_w=self.W, face_w=self.H // 2)
        self.frame_count = 0

        # 点群用: 方向ベクトル格子（入力解像度、ダウンサンプル考慮）
        ds = max(self.pcl_downsample, 1)
        h_ds = self.input_h // ds
        w_ds = self.input_w // ds
        u = (np.arange(w_ds) * ds + 0.5) / self.input_w * 2 * np.pi - np.pi
        v = np.pi / 2 - (np.arange(h_ds) * ds + 0.5) / self.input_h * np.pi
        uu, vv = np.meshgrid(u, v)
        x = np.cos(vv) * np.sin(uu)
        y = np.sin(vv)
        z = np.cos(vv) * np.cos(uu)
        self.dirs = np.stack((x, y, z), axis=2)

        self.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.INT32,   count=1),
        ]
        self.point_dtype = np.dtype([('x', np.float32), ('y', np.float32),
                                     ('z', np.float32), ('rgb', np.int32)])

        self.get_logger().info(
            f'Multi-camera node initialized: {self.num_cameras} cams, '
            f'model {self.H}x{self.W}, backend={backend_name}')

    def cb_image(self, idx, msg, trigger):
        self.latest_msgs[idx] = msg
        if not trigger:
            return
        if any(m is None for m in self.latest_msgs):
            return

        self.frame_count += 1
        if self.frame_skip > 1:
            if self.frame_count % self.frame_skip != 0:
                return
            self.frame_count = 0

        self.process()

    def process(self):
        # 1) 前処理: 入力解像度リサイズ → モデル解像度 → 正規化バッチ + cube ストリップ
        imgs_input = []
        equi_imgs = []
        cube_imgs = []
        for m in self.latest_msgs:
            img = self.br.imgmsg_to_cv2(m, desired_encoding='rgb8')
            img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            imgs_input.append(img)
            if self.input_h != self.H or self.input_w != self.W:
                img_model = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            else:
                img_model = img
            equi_imgs.append(img_model)
            cube_imgs.append(self.E2C.run(img_model))

        equi_np = normalize_batch(equi_imgs)
        cube_np = normalize_batch(cube_imgs)

        # 2) バッチ推論 → (B,H,W)
        depth_batch = self.backend.infer(equi_np, cube_np)

        # 3) カメラごとに後処理・パブリッシュ
        stamp = self.get_clock().now().to_msg()
        for i in range(self.num_cameras):
            depth = depth_batch[i]
            if self.input_h != self.H or self.input_w != self.W:
                depth = cv2.resize(depth, (self.input_w, self.input_h),
                                   interpolation=cv2.INTER_LINEAR)
            depth = self.smoothers[i].apply(depth)

            # pred_depth は距離とともに増加する(相対)深度。逆数を取ると
            # シーンが半径方向に反転する（実画像で検証済み）。
            # 絶対距離への変換 d=(r-β)/α はビューワ側の床基準アフィン較正が行う。
            radius = (self.scale_factor * depth).astype(np.float32)

            header = self.latest_msgs[i].header
            header.frame_id = 'camera_link'
            header.stamp = stamp

            if self.publish_depth:
                depth_msg = self.br.cv2_to_imgmsg(radius, encoding='32FC1')
                depth_msg.header = header
                self.pubs_depth[i].publish(depth_msg)

            if self.publish_pointcloud:
                self.publish_cloud(i, radius, imgs_input[i], header)

    def publish_cloud(self, idx, radius, img_rgb, header):
        ds = max(self.pcl_downsample, 1)
        radius_ds = radius[::ds, ::ds]
        img_ds = img_rgb[::ds, ::ds]

        pts = (self.dirs * radius_ds[..., None]).reshape(-1, 3)
        colors_img = img_ds.reshape(-1, 3)
        colors = (colors_img[:, 0].astype(np.int32) << 16) | \
                 (colors_img[:, 1].astype(np.int32) << 8) | \
                 colors_img[:, 2].astype(np.int32)

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
        self.pubs_points[idx].publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = DepthAnywhereMulti()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
