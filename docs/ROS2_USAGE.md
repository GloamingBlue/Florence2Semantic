# Florence2 ROS2 节点使用说明
> 关于Qwen3 ROS2 节点，使用方法相同

## 一、依赖安装

### 必需依赖
```bash
pip install torch transformers pillow numpy
```

### ROS2 相关依赖
```bash
# ROS2 Python 客户端库（通常随 ROS2 安装）
# 如果没有，可以尝试：
pip install rclpy

# cv_bridge（用于图像消息转换）
# Ubuntu/Debian:
sudo apt-get install ros-<distro>-cv-bridge

# 或从源码编译安装
```

## 二、使用方式

### 方式 1: 命令行模式（测试用）

```bash
python code/florence2_caption_ros2.py \
    --image /path/to/image.jpg \
    --model_path /home/ubun/xanylabeling_data/models/florence \
    --task_type caption \
    --show_timing
```

### 方式 2: ROS2 节点模式

#### 2.1 使用 YAML 配置文件（推荐）

```bash
# 激活 ROS2 环境
source /opt/ros/<distro>/setup.bash

# 使用 YAML 配置文件运行节点
python code/florence2_caption_ros2.py --ros2 \
    --ros-args --params-file florence/florence2_caption_params.yaml
```

配置文件 `florence2_caption_params.yaml` 包含所有可配置参数，推荐使用此方式。

#### 2.2 直接运行（使用默认参数）

```bash
# 激活 ROS2 环境
source /opt/ros/<distro>/setup.bash

# 运行节点（使用代码中的默认参数）
python code/florence2_caption_ros2.py --ros2
```

#### 2.3 使用 ros2 run（需要安装为包）

```bash
ros2 run florence2_caption florence2_caption_ros2 \
    --ros-args --params-file florence/florence2_caption_params.yaml
```

#### 2.4 命令行参数覆盖（临时修改）

```bash
# 使用 YAML 配置文件，同时覆盖某些参数
python code/florence2_caption_ros2.py --ros2 \
    --ros-args \
    --params-file configs/florence2_caption_params.yaml \
    -p image_topic:=/camera/camera/color/image_raw \
    -p task_type:=caption
```

注意：命令行参数会覆盖 YAML 文件中的配置。

## 三、图像源配置

节点支持两种图像获取方式：

### 3.1 ROS2 话题模式（默认）

从 ROS2 话题订阅图像消息。

**配置参数**：
- `image_source`: `"ros2"`（默认）
- `image_topic`: 图像话题名称（默认: `/camera/camera/color/image_raw`）

### 3.2 RTSP 流模式

从 RTSP 视频流获取图像。

**配置参数**：
- `image_source`: `"rtsp"`
- `rtsp_url`: RTSP 流地址（默认: `rtsp://192.168.168.168:8554/test`）

**特点**：
- 不需要启动相机 ROS2 节点
- 支持网络视频流
- 自动重连机制
- 在独立线程中读取，不阻塞主线程

## 四、ROS2 话题

### 订阅话题

1. **图像话题** (仅在 `image_source="ros2"` 时使用)
   - 默认: `/camera/camera/color/image_raw`
   - 类型: `sensor_msgs/Image`
   - 用途: 持续接收图像流，保存最新一帧
   - 可在 YAML 配置文件中修改

2. **控制信号话题** (默认: `/navigation/florence`)
   - 类型: `std_msgs/Int8`
   - 用途: 接收处理触发信号
   - 值: `0` = 不处理, `1` = 处理当前最新帧
   - 可在 YAML 配置文件中修改

### 发布话题

1. **结果话题** (默认: `/florence2/caption`)
   - 类型: `std_msgs/String`
   - 内容: 生成的图像描述文本
   - 可在 YAML 配置文件中修改

## 五、测试步骤

### 方式 A: ROS2 话题模式

#### 步骤 0: 启动 RealSense ROS2 节点

```bash
# 终端 0
source /path_to_your_realsense_ros2_ws/install/setup.zsh
ros2 launch realsense2_camera rs_launch.py
```

#### 步骤 1: 配置并启动节点

确保配置文件中 `image_source: "ros2"`：

```yaml
florence2_control_node:
  ros__parameters:
    image_source: "ros2"
    image_topic: "/camera/camera/color/image_raw"
    # ... 其他参数
```

启动节点：
```bash
# 终端 1
python code/florence2_caption_ros2.py --ros2 --ros-args --params-file configs/florence2_caption_params.yaml
```

预期输出：
```
📷 已订阅图像话题: /camera/camera/color/image_raw
🎮 已订阅控制信号话题: /navigation/florence
📤 已创建结果发布话题: /florence2/caption
✅ Florence2 Caption Node 初始化完成（模型未加载）
⏳ 等待控制信号...
```

### 方式 B: RTSP 流模式

#### 步骤 0: 确保 RTSP 流可用

```bash
# 测试 RTSP 流是否可访问（可选）
ffplay rtsp://192.168.168.168:8554/test
# 或
vlc rtsp://192.168.168.168:8554/test
```

#### 步骤 1: 配置并启动节点

确保配置文件中 `image_source: "rtsp"`：

```yaml
florence2_control_node:
  ros__parameters:
    image_source: "rtsp"
    rtsp_url: "rtsp://192.168.168.168:8554/test"
    # ... 其他参数
```

启动节点：
```bash
# 终端 1
python code/florence2_caption_ros2.py --ros2 --ros-args --params-file configs/florence2_caption_params.yaml
```

预期输出：
```
🔄 正在连接 RTSP 流: rtsp://192.168.168.168:8554/test
🔄 正在验证 RTSP 流连接...
✅ RTSP 流连接验证成功，已读取第一帧 (尺寸: (480, 640, 3))
✅ RTSP 流读取线程已启动: rtsp://192.168.168.168:8554/test
🎮 已订阅控制信号话题: /navigation/florence
📤 已创建结果发布话题: /florence2/caption
✅ Florence2 Caption Node 初始化完成（模型未加载）
⏳ 等待控制信号...
```

**注意**：RTSP 模式不需要启动相机 ROS2 节点。

#### 步骤 2: 发送控制信号（触发处理）

```bash
# 终端 2
ros2 topic pub -1 /nav/arrival std_msgs/msg/String "{data: '操场'}"
ros2 topic pub -1 /navigation/florence std_msgs/Int8 "data: 1"
```

#### 步骤 3: 查看结果

```bash
# 终端 3
ros2 topic echo -f /florence2/caption
```

#### 步骤 4: 查看节点日志

节点会在终端 1 输出处理日志：

**ROS2 话题模式**：
```
收到控制信号 1: 开始处理图像...
🔄 转换图像...
🔄 正在加载 Florence2 模型（按需加载）...
✅ 生成描述: A person is standing in front of a building...
📤 已发布描述结果
```

**RTSP 流模式**：
```
收到控制信号 1: 开始处理图像...
🔄 转换图像...
🔄 正在加载 Florence2 模型（按需加载）...
✅ 生成描述: A person is standing in front of a building...
📤 已发布描述结果
```

## 六、配置参数

### 6.1 配置文件方式（推荐）

所有参数都可以通过 YAML 配置文件 `florence2_caption_params.yaml` 进行配置：

**ROS2 话题模式配置示例**：
```yaml
florence2_control_node:
  ros__parameters:
    # 图像源配置
    image_source: "ros2"
    image_topic: "/camera/camera/color/image_raw"
    
    # 控制信号话题
    control_topic: "/navigation/florence"
    
    # 模型相关配置
    model_path: "/home/ubun/xanylabeling_data/models/florence"
    task_type: "more_detailed_cap"
    
    # 结果发布话题
    result_topic: "/florence2/caption"
    
    # 其他参数...
```

**RTSP 流模式配置示例**：
```yaml
florence2_control_node:
  ros__parameters:
    # 图像源配置
    image_source: "rtsp"
    rtsp_url: "rtsp://192.168.168.168:8554/test"
    
    # 控制信号话题
    control_topic: "/navigation/florence"
    
    # 模型相关配置
    model_path: "/home/ubun/xanylabeling_data/models/florence"
    task_type: "more_detailed_cap"
    
    # 结果发布话题
    result_topic: "/florence2/caption"
    
    # 其他参数...
```

使用配置文件启动：
```bash
python code/florence2_caption_ros2.py --ros2 \
    --ros-args --params-file configs/florence2_caption_params.yaml
```

### 6.2 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| **图像源配置** |
| `image_source` | string | `"ros2"` | 图像来源: `"ros2"` 或 `"rtsp"` |
| `image_topic` | string | `/camera/camera/color/image_raw` | ROS2 图像话题（仅在 `image_source="ros2"` 时使用） |
| `rtsp_url` | string | `rtsp://192.168.168.168:8554/test` | RTSP 流地址（仅在 `image_source="rtsp"` 时使用） |
| **控制与结果** |
| `control_topic` | string | `/navigation/florence` | 控制信号话题 |
| `result_topic` | string | `/florence2/caption` | 结果发布话题 |
| **模型配置** |
| `model_path` | string | `/home/ubun/.../florence` | 模型路径 |
| `task_type` | string | `more_detailed_cap` | 任务类型: `caption`, `detailed_cap`, `more_detailed_cap` |
| `max_new_tokens` | int | `1024` | 最大生成 token 数 |
| `num_beams` | int | `3` | Beam search 的 beam 数量 |
| `do_sample` | bool | `false` | 是否使用采样生成 |
| `trust_remote_code` | bool | `true` | 是否信任远程代码 |
| **翻译配置** |
| `translate_to_chinese` | bool | `false` | 是否将生成的英文描述翻译为中文 |
| `translation_model` | string | `Helsinki-NLP/opus-mt-en-zh` | 翻译模型（HuggingFace ID） |
| `translation_model_path` | string | `""` | 翻译模型本地路径（可选） |
| **其他配置** |
| `show_timing` | bool | `true` | 是否在日志中显示时间统计 |
| `flip` | bool | `false` | 是否在语义生成前将图像旋转180度 |

### 6.3 命令行参数覆盖

如果需要临时修改某些参数，可以在命令行中覆盖：

```bash
python code/florence2_caption_ros2.py --ros2 \
    --ros-args \
    --params-file configs/florence2_caption_params.yaml \
    -p image_source:=rtsp \
    -p rtsp_url:=rtsp://192.168.1.100:8554/stream \
    -p task_type:=caption \
    -p show_timing:=true
```

命令行参数会覆盖 YAML 文件中的配置。

## 七、常见问题

### 问题 1: ROS2 不可用

**错误**: `ROS2 不可用，请安装: pip install rclpy`

**解决**: 
- 确保已安装 ROS2
- 激活 ROS2 环境: `source /opt/ros/<distro>/setup.bash`
- 如果使用 conda 环境，可能需要安装: `pip install rclpy`

### 问题 2: cv_bridge 不可用

**错误**: `cv_bridge 不可用，无法转换图像消息`

**解决**:
```bash
# Ubuntu/Debian
sudo apt-get install ros-<distro>-cv-bridge

# 或从源码编译
```

### 问题 3: 没有收到图像（ROS2 模式）

**现象**: 发送控制信号后，节点提示"尚未收到图像"

**解决**:
- 检查图像话题是否正确: `ros2 topic list`
- 检查图像话题是否有数据: `ros2 topic echo /camera/color/image_raw`
- 确认 RealSense 相机节点正在运行
- 确认配置文件中 `image_source: "ros2"`

### 问题 3b: 没有收到 RTSP 帧

**现象**: 发送控制信号后，节点提示"尚未收到 RTSP 帧"

**解决**:
- 检查 RTSP 流地址是否正确: `rtsp://192.168.168.168:8554/test`
- 测试 RTSP 流是否可访问: `ffplay rtsp://192.168.168.168:8554/test`
- 检查网络连接: `ping 192.168.168.168`
- 确认配置文件中 `image_source: "rtsp"`
- 查看节点启动日志，确认 RTSP 流连接是否成功
- 等待几秒让 RTSP 流稳定后再发送控制信号

### 问题 4: 处理速度慢

**现象**: 处理一张图像需要很长时间

**解决**:
- 使用 GPU 加速（确保 CUDA 可用）
- 降低 `max_new_tokens` 参数
- 使用 `caption` 而不是 `more_detailed_cap`

## 八、与导航系统集成

### 发送控制信号

在导航系统中，当需要生成图像描述时，发布信号：

```python
# Python 示例
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        self.control_pub = self.create_publisher(
            Int8, 
            '/navigation/florence', 
            10
        )
    
    def trigger_caption(self):
        """触发图像描述生成"""
        msg = Int8()
        msg.data = 1
        self.control_pub.publish(msg)
        self.get_logger().info('已发送处理信号')
```

### 接收结果

```python
# Python 示例
from std_msgs.msg import String

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        self.caption_sub = self.create_subscription(
            String,
            '/florence2/caption',
            self.caption_callback,
            10
        )
    
    def caption_callback(self, msg: String):
        """接收图像描述"""
        caption = msg.data
        self.get_logger().info(f'收到图像描述: {caption}')
        # 处理描述结果...
```

## 九、性能优化建议

1. **使用 GPU**: 确保 CUDA 可用，模型会自动使用 GPU
2. **调整任务类型**: `caption` 比 `more_detailed_cap` 快
3. **调整生成参数**: 降低 `max_new_tokens` 和 `num_beams`
4. **避免频繁触发**: 控制信号发送频率不要过高

## 十、日志级别

节点使用 ROS2 日志系统，可以通过环境变量控制日志级别：

```bash
# 设置日志级别为 DEBUG（更详细）
export RCUTILS_LOGGING_SEVERITY=DEBUG

# 设置日志级别为 INFO（默认）
export RCUTILS_LOGGING_SEVERITY=INFO

# 设置日志级别为 WARN（只显示警告和错误）
export RCUTILS_LOGGING_SEVERITY=WARN
```

