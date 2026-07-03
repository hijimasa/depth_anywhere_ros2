# depth_anywhere_ros2

ROS 2 package for monocular depth estimation using Depth-Anywhere. This package provides ROS 2 interfaces for the [Depth-Anywhere](https://github.com/albert100121/Depth-Anywhere).

## Acknowledgment

This package is based on the [Depth-Anywhere](https://github.com/albert100121/Depth-Anywhere). The original implementation and research are credited to the authors of that work.

## Features

- Convert RGB images to depth maps using Depth-Anywhere
- ROS 2 interfaces for easy integration with other robotics components

## Installation

### Prerequisites

- ROS 2 (tested with Humble)
- CUDA-capable GPU (recommended)
- PyTorch
- OpenCV

### Steps

1. Clone this repository to your ROS 2 workspace:
    ```bash
    cd ~/your_ros2_ws/src
    git clone https://github.com/yourusername/depth_anywhere_ros2.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Build the package:
    ```bash
    cd ~/your_ros2_ws
    colcon build --packages-select depth_anywhere_ros2
    ```

4. Source the workspace:
    ```bash
    source ~/your_ros2_ws/install/setup.bash
    ```

## Usage

### Launch the node (single camera)

```bash
ros2 run depth_anywhere_ros2 infer --ros-args -r image:=/image_topic -r points:=/points -p model_name:=UniFuse -p device:=cuda
```

### Launch the node (multi camera / batch inference)

複数カメラを1ノードで batch 推論する場合は `infer_multi` を使う（GPU利用効率が良い）。

```bash
ros2 run depth_anywhere_ros2 infer_multi --ros-args \
    -r image_0:=/theta1_image -r image_1:=/theta2_image \
    -r depth_0:=/theta1_depth -r depth_1:=/theta2_depth
```

### Parameters

- `model_name`: Depth-Anywhere model variant to use (`UniFuse`, `BiFuseV2`, `HoHoNet`, ``)
- `device`: Whether to use acceleration device (default: `cuda`)
- `equi_h` / `equi_w`: モデルの推論解像度。学習時の 512x1024 以外でも動く（256x512 で約4倍高速）
- `publish_depth`: 半径マップ (`sensor_msgs/Image`, 32FC1) を配信する（`tkg_tps_viewer_gl` の depth モード用）
- `publish_pointcloud`: 従来の `sensor_msgs/PointCloud2` を配信する
- `backend` (infer_multi のみ): `pytorch`（デフォルト）または `tensorrt`
- `trt_engine_path` (infer_multi のみ): TensorRT エンジンファイルのパス

### Topics

- **Subscribed**: image (`sensor_msgs/Image`)
- **Published**:
  - points (`sensor_msgs/PointCloud2`)
  - depth (`sensor_msgs/Image`, 32FC1): 各画素の視線方向に掛ける半径 [m]（`scale_factor / モデル出力`）

## TensorRT engine (optional)

`infer_multi` の `backend: tensorrt` を使うには、**事前に一度だけ手動で**エンジンを作成する必要がある。
ノード起動時に自動でエクスポートは行われない（エンジンのビルドは数分かかるうえ、GPU機種ごとに固有のため）。

作成手順（**手順2は必ず実行する実機＝Jetson 上で行うこと**。エンジンは GPU 機種・TensorRT バージョンに依存し、他のマシンで作ったものは使えない。手順1の ONNX エクスポートだけは開発PCでも可）:

```bash
# 1. ONNX エクスポート（PyTorch モデル → ONNX。バッチサイズはカメラ台数に合わせる）
cd <this package>
python3 scripts/export_unifuse_onnx.py --height 256 --width 512 --batch 2

# 2. TensorRT エンジンのビルド（Jetson 実機上で実行、数分かかる）
/usr/src/tensorrt/bin/trtexec --onnx=unifuse_256x512_b2.onnx \
    --saveEngine=/path/to/unifuse_256x512_b2_fp16.engine --fp16
```

作成したエンジンをノードに渡す:

```bash
ros2 run depth_anywhere_ros2 infer_multi --ros-args \
    -p backend:=tensorrt \
    -p trt_engine_path:=/path/to/unifuse_256x512_b2_fp16.engine \
    -p equi_h:=256 -p equi_w:=512
```

注意:
- エンジンの形状（バッチ数・解像度）とノードのパラメータ（`num_cameras`, `equi_h`, `equi_w`）が
  一致しない場合、起動時にエラーメッセージを出して停止する。
- 解像度やバッチ数を変えた場合はエンジンの作り直しが必要。
- JetPack を更新（TensorRT のバージョンが変わる）した場合もエンジンの作り直しが必要。

## License

This project is licensed under the same terms as the original Depth-Anywhere project. Please refer to the original repository for licensing information.

## References

- [Depth-Anywhere](https://github.com/albert100121/Depth-Anywhere): The original implementation this package is based on.