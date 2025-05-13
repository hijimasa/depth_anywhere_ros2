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

### Launch the node

```bash
ros2 run depth_anywhere_ros2 infer --ros-args -p model_name:=UniFuse -p image_topic:=/image_topic -p points_topic:=/points -p device:=cuda
```

### Parameters

- `image_topic`: Input RGB image topic (default: `image`)
- `points_topic`: Output Pointcloud topic (default: `points`)
- `model_name`: Depth-Anywhere model variant to use
- `device`: Whether to use acceleration device (default: `cuda`)

### Topics

- **Subscribed**: RGB image (`sensor_msgs/Image`)
- **Published**: PointCloud (`sensor_msgs/PointCloud2`)

## License

This project is licensed under the same terms as the original Depth-Anywhere project. Please refer to the original repository for licensing information.

## References

- [Depth-Anywhere](https://github.com/albert100121/Depth-Anywhere): The original implementation this package is based on.