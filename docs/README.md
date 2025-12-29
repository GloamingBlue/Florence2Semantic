# Florence2Semantic

This project is used for video semantic generation of the RealSense camera on the mechanical guide dog.

## 文档

- [详细文档](https://www.yuque.com/u57806034/mdfxam/cobhfh7nmiphlgbg)
- [ROS2 使用说明](https://github.com/GloamingBlue/Florence2Semantic/blob/main/docs/ROS2_USAGE.md)

## 文件说明

### 核心脚本

- **`florence2_test.py`**: 语义分析功能的最小实现，适用于分析本地图像进行测试
- **`florence2_caption_ros2.py`**: 将前者集成到 ROS2 中，实时监测视频流接收控制信号进行语义分析，支持：
  - ROS2 话题模式：从 ROS2 话题订阅图像
  - RTSP 流模式：从 RTSP 视频流获取图像
  - 翻译功能：将英文描述翻译为中文
- **`qwen3_test.py`**: qwen3-vl语义分析功能的最小实现，适用于分析本地图像进行测试
- **`qwen3_caption_ros2.py`**: 将前者集成到 ROS2 中，实时监测视频流接收控制信号进行语义分析，支持：
  - ROS2 话题模式：从 ROS2 话题订阅图像
  - RTSP 流模式：从 RTSP 视频流获取图像

### 配置文件

- **`configs/florence2_caption_params.yaml`**: ROS2 节点的配置文件，支持：
  - 图像源选择（ROS2 话题或 RTSP 流）
  - 模型参数配置
  - 翻译功能配置
  - 兼容命令行参数覆盖
- **`configs/qwen3vl_caption_params.yaml`**: ROS2 节点的配置文件，支持：
  - 图像源选择（ROS2 话题或 RTSP 流）
  - 模型参数配置
  - 兼容命令行参数覆盖

## 主要功能

### 图像源支持

1. **ROS2 话题模式**（默认）
   - 从 ROS2 话题订阅图像消息
   - 适用于 RealSense 相机等 ROS2 设备

2. **RTSP 流模式**
   - 从 RTSP 视频流获取图像
   - 支持网络视频流
   - 自动重连机制
   - 不需要启动相机 ROS2 节点

### 任务类型

- `caption`: 基础图像描述
- `detailed_cap`: 详细图像描述
- `more_detailed_cap`: 更详细的图像描述

### 其他特性

- 按需加载模型，节省内存
- 支持中英文翻译
- 支持 GPU 加速
- 线程安全的图像处理
- 详细的错误处理和日志

## 快速开始

### ROS2 话题模式

```bash
# 1. 启动相机节点
source /path_to_your_realsense_ros2_ws/install/setup.zsh
ros2 launch realsense2_camera rs_launch.py

# 2. 启动语义分析节点
python code/florence2_caption_ros2.py --ros2 --ros-args --params-file configs/florence2_caption_params.yaml
或
python code/qwen3_caption_ros2.py --ros2 --ros-args --params-file configs/qwen3vl_params.yaml

# 3. 发送控制信号
ros2 topic pub -1 /nav/arrival std_msgs/msg/String "{data: '操场'}"
ros2 topic pub -1 /navigation/florence std_msgs/Int8 "data: 1"

# 4. 查看结果
ros2 topic echo -f /florence2/caption
```

### RTSP 流模式

```yaml
# 在 configs/florence2_caption_params.yaml 中设置
image_source: "rtsp"
rtsp_url: "rtsp://192.168.168.168:8554/test"
```

```bash
# 1. 启动语义分析节点（不需要启动相机节点）
python code/florence2_caption_ros2.py --ros2 --ros-args --params-file configs/florence2_caption_params.yaml
或
python code/qwen3_caption_ros2.py --ros2 --ros-args --params-file configs/qwen3vl_params.yaml

# 2. 发送控制信号
ros2 topic pub -1 /nav/arrival std_msgs/msg/String "{data: '操场'}"
ros2 topic pub -1 /navigation/florence std_msgs/Int8 "data: 1"

# 3. 查看结果
ros2 topic echo -f /florence2/caption
```

详细使用说明请参考 [ROS2_USAGE.md](docs/ROS2_USAGE.md)。  
