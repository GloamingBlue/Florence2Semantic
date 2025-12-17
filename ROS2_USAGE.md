# Florence2 ROS2 节点使用说明

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
python florence/florence2_caption_ros2.py \
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
python florence/florence2_caption_ros2.py --ros2 \
    --ros-args --params-file florence/florence2_caption_params.yaml
```

配置文件 `florence2_caption_params.yaml` 包含所有可配置参数，推荐使用此方式。

#### 2.2 直接运行（使用默认参数）

```bash
# 激活 ROS2 环境
source /opt/ros/<distro>/setup.bash

# 运行节点（使用代码中的默认参数）
python florence/florence2_caption_ros2.py --ros2
```

#### 2.3 使用 ros2 run（需要安装为包）

```bash
ros2 run florence2_caption florence2_caption_ros2 \
    --ros-args --params-file florence/florence2_caption_params.yaml
```

#### 2.4 命令行参数覆盖（临时修改）

```bash
# 使用 YAML 配置文件，同时覆盖某些参数
python florence/florence2_caption_ros2.py --ros2 \
    --ros-args \
    --params-file florence/florence2_caption_params.yaml \
    -p image_topic:=/camera/camera/color/image_raw \
    -p task_type:=caption
```

注意：命令行参数会覆盖 YAML 文件中的配置。

## 三、ROS2 话题

### 订阅话题

1. **图像话题** (默认: `/camera/camera/color/image_raw`)
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

## 四、测试步骤

### 步骤 1: 启动节点

```bash
# 终端 1 - 使用 YAML 配置文件（推荐）
python florence/florence2_caption_ros2.py --ros2 --ros-args --params-file florence/configs/florence2_caption_params.yaml
```

预期输出：
```
🔄 正在初始化 Florence2 模型...
✅ 模型加载完成
📷 已订阅图像话题: /camera/camera/color/image_raw
🎮 已订阅控制信号话题: /navigation/florence
📤 已创建结果发布话题: /florence2/caption
✅ Florence2 Caption Node 初始化完成
⏳ 等待控制信号...
```

### 步骤 2: 发送控制信号（触发处理）

```bash
# 终端 2
ros2 topic pub -1 /navigation/florence std_msgs/Int8 "data: 1"
```

### 步骤 3: 查看结果

```bash
# 终端 3
ros2 topic echo -f /florence2/caption
```

### 步骤 4: 查看节点日志

节点会在终端 1 输出处理日志：
```
收到控制信号 1: 开始处理图像...
✅ 生成描述: A person is standing in front of a building...
📤 已发布描述结果
```

## 五、配置参数

### 5.1 配置文件方式（推荐）

所有参数都可以通过 YAML 配置文件 `florence2_caption_params.yaml` 进行配置：

```yaml
florence2_caption_node:
  ros__parameters:
    image_topic: "/camera/camera/color/image_raw"
    control_topic: "/navigation/florence"
    model_path: "/home/ubun/xanylabeling_data/models/florence"
    task_type: "more_detailed_cap"
    result_topic: "/florence2/caption"
    show_timing: false
    max_new_tokens: 1024
    num_beams: 3
    do_sample: false
    trust_remote_code: true
```

使用配置文件启动：
```bash
python florence/florence2_caption_ros2.py --ros2 \
    --ros-args --params-file florence/florence2_caption_params.yaml
```

### 5.2 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `image_topic` | string | `/camera/camera/color/image_raw` | 图像流话题 |
| `control_topic` | string | `/navigation/florence` | 控制信号话题 |
| `model_path` | string | `/home/ubun/xanylabeling_data/models/florence` | 模型路径 |
| `task_type` | string | `more_detailed_cap` | 任务类型: caption, detailed_cap, more_detailed_cap |
| `result_topic` | string | `/florence2/caption` | 结果发布话题 |
| `show_timing` | bool | `false` | 是否在日志中显示时间统计 |
| `max_new_tokens` | int | `1024` | 最大生成 token 数 |
| `num_beams` | int | `3` | Beam search 的 beam 数量 |
| `do_sample` | bool | `false` | 是否使用采样生成 |
| `trust_remote_code` | bool | `true` | 是否信任远程代码 |

### 5.3 命令行参数覆盖

如果需要临时修改某些参数，可以在命令行中覆盖：

```bash
python florence/florence2_caption_ros2.py --ros2 \
    --ros-args \
    --params-file florence/florence2_caption_params.yaml \
    -p task_type:=caption \
    -p show_timing:=true
```

命令行参数会覆盖 YAML 文件中的配置。

## 六、常见问题

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

### 问题 3: 没有收到图像

**现象**: 发送控制信号后，节点提示"尚未收到图像"

**解决**:
- 检查图像话题是否正确: `ros2 topic list`
- 检查图像话题是否有数据: `ros2 topic echo /camera/color/image_raw`
- 确认 RealSense 相机节点正在运行

### 问题 4: 处理速度慢

**现象**: 处理一张图像需要很长时间

**解决**:
- 使用 GPU 加速（确保 CUDA 可用）
- 降低 `max_new_tokens` 参数
- 使用 `caption` 而不是 `more_detailed_cap`

## 七、与导航系统集成

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

## 八、性能优化建议

1. **使用 GPU**: 确保 CUDA 可用，模型会自动使用 GPU
2. **调整任务类型**: `caption` 比 `more_detailed_cap` 快
3. **调整生成参数**: 降低 `max_new_tokens` 和 `num_beams`
4. **避免频繁触发**: 控制信号发送频率不要过高

## 九、日志级别

节点使用 ROS2 日志系统，可以通过环境变量控制日志级别：

```bash
# 设置日志级别为 DEBUG（更详细）
export RCUTILS_LOGGING_SEVERITY=DEBUG

# 设置日志级别为 INFO（默认）
export RCUTILS_LOGGING_SEVERITY=INFO

# 设置日志级别为 WARN（只显示警告和错误）
export RCUTILS_LOGGING_SEVERITY=WARN
```

