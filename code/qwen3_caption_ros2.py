#!/usr/bin/env python3
"""
Qwen3-VL 图像描述生成脚本（精简版）
使用 Qwen3-VL-2B-Instruct 模型生成图像描述（HuggingFace 格式）
精简版：移除了性能监测和时间计算功能，只保留核心语义生成功能
"""

import warnings
import sys
import argparse
import threading
import gc
from pathlib import Path
from unittest.mock import patch
from typing import Union, Optional

warnings.filterwarnings("ignore")

try:
    import torch
    from PIL import Image
    import numpy as np
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from transformers.dynamic_module_utils import get_imports
except ImportError as e:
    print(f"❌ 缺少必要的依赖包: {e}")
    print("请安装: pip install torch transformers pillow numpy")
    sys.exit(1)

# 导入 ROS2 相关库
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Int8, String, Bool

# 导入 cv_bridge 和 OpenCV，用于图像转换
from cv_bridge import CvBridge
import cv2


class Qwen3VLCaption:
    """Qwen3-VL 图像描述生成器"""

    def __init__(
        self,
        model_path: str,
        prompt_template: str,
        trust_remote_code: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        do_sample: bool = True,
    ):
        """
        初始化 Qwen3-VL 模型（HuggingFace 格式）

        Args:
            model_path: 模型路径（本地路径或 HuggingFace 模型 ID，如 "Qwen/Qwen3-VL-2B-Instruct"）
            prompt_template: 提示词模板（从配置文件读取）
            trust_remote_code: 是否信任远程代码
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
            do_sample: 是否使用采样生成
        """
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # 自动选择设备
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

        print(f"🔄 正在加载模型: {model_path}")
        print(f"📱 使用设备: {self.device}")

        # 修复 CPU 上 flash_attn 的问题
        def fixed_get_imports(filename):
            imports = get_imports(filename)
            if not torch.cuda.is_available() and "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports

        # 加载模型和处理器
        with patch(
            "transformers.dynamic_module_utils.get_imports", fixed_get_imports
        ):
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=trust_remote_code,
                attn_implementation="eager",  # 因为aarch架构上暂未找到适配gqa的torch版本,如果用的是x86架构,可以注释掉这行
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )

        print(f"✅ Qwen3-VL 模型加载完成")

    def generate_caption(
        self, 
        image: Union[str, Image.Image, np.ndarray]
    ) -> str:
        """
        为图像生成描述

        Args:
            image: 图像输入，可以是：
                  - str: 图像文件路径
                  - PIL.Image: PIL 图像对象
                  - np.ndarray: numpy 数组（RGB 格式，shape: [H, W, 3]）

        Returns:
            图像描述文本
        """
        # 读取和转换图像
        if isinstance(image, str):
            # 文件路径
            if not Path(image).exists():
                raise FileNotFoundError(f"图像文件不存在: {image}")
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # PIL.Image
            pil_image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            # numpy 数组
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"numpy 数组必须是 RGB 格式，shape: [H, W, 3]，当前: {image.shape}")
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}，支持类型: str, PIL.Image, np.ndarray")

        # 构建消息格式（根据官方文档）
        prompt = self.prompt_template
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,  # 直接传入 PIL Image 对象
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 使用 apply_chat_template 处理消息（根据官方文档）
        print("🤖 正在生成描述...")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 将输入移动到设备
        inputs = inputs.to(self.device)

        # 生成描述
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                repetition_penalty=1.3,  # 添加重复惩罚，减少重复生成（值越大，惩罚越强）
            )

        # 截取新生成的部分（根据官方文档）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 解码生成的文本
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # 提取描述文本
        final_caption = output_text[0].strip() if output_text else ""

        return final_caption
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ROS2 轻量级控制节点（不加载模型）
class Qwen3VLControlNode(Node):
    """
    轻量级控制节点，负责：
    - 持续接收图像流，保存最新一帧
    - 监听控制信号
    - 收到信号 1 时，按需加载模型、处理图像、释放资源
    """
    
    def __init__(self):
        super().__init__('qwen3vl_control_node')
        
        # 参数声明
        self.declare_parameter('image_source', 'ros2')  # 图像来源: "ros2" 或 "rtsp"
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')  # ROS2 图像话题
        self.declare_parameter('rtsp_url', 'rtsp://192.168.168.168:8554/test')  # RTSP 流地址
        self.declare_parameter('control_topic', '/navigation/florence')  # 控制信号话题 1 (String类型，触发词: "操场")
        self.declare_parameter('control_topic_2', '/nav/arrival')  # 控制信号话题 2 (Int8类型，期望值: 1或2，触发发送)
        self.declare_parameter('model_path', 'Qwen/Qwen3-VL-2B-Instruct')  # Qwen3-VL 模型路径（本地路径或 HuggingFace ID）
        self.declare_parameter('caption_prompt', '')  # caption 提示词模板（值为1时使用）
        self.declare_parameter('text_cap_prompt', '')  # text_cap 提示词模板（值为2时使用）
        self.declare_parameter('result_topic', '/florence2/caption')
        self.declare_parameter('max_new_tokens', 1024)
        self.declare_parameter('temperature', 0.7)  # 采样温度
        self.declare_parameter('top_p', 0.8)  # nucleus sampling
        self.declare_parameter('do_sample', True)  # 是否使用采样生成
        self.declare_parameter('trust_remote_code', True)  # 是否信任远程代码
        self.declare_parameter('flip', False)  # 是否在语义生成前将图像旋转180度
        
        # 获取图像源类型
        image_source = self.get_parameter('image_source').value
        
        # 线程安全：最新图像存储
        self.latest_image_lock = threading.Lock()
        self.latest_image_msg = None  # ROS2 图像消息
        self.latest_rtsp_frame = None  # RTSP 帧（numpy array）
        
        # 处理状态标志（避免重复处理）
        self.is_processing = False
        self.processing_lock = threading.Lock()
        
        # 结果缓存（分别缓存不同 prompt 的结果）
        self.cached_caption_result = None  # caption_prompt 的缓存结果
        self.cached_text_cap_result = None  # text_cap_prompt 的缓存结果
        self.cache_lock = threading.Lock()
        
        # RTSP 相关
        self.rtsp_cap = None
        self.rtsp_thread = None
        self.rtsp_running = False
        
        # 初始化 cv_bridge（用于图像转换）
        self.cv_bridge = CvBridge()
        
        # 根据图像源类型初始化
        if image_source == 'ros2':
            # ROS2 模式：创建图像订阅者
            image_topic = self.get_parameter('image_topic').value
            self.image_subscription = self.create_subscription(
                ROSImage,
                image_topic,
                self.image_callback,
                1  # QoS depth = 1，只保留最新图像
            )
            self.get_logger().info(f'📷 已订阅图像话题: {image_topic}')
        elif image_source == 'rtsp':
            # RTSP 模式：启动 RTSP 流读取线程
            rtsp_url = self.get_parameter('rtsp_url').value
            self._start_rtsp_stream(rtsp_url)
        else:
            raise ValueError(f'不支持的图像源类型: {image_source}，支持的类型: "ros2", "rtsp"')
        
        # 创建控制信号订阅者（String 类型，接收 "操场" 等触发词）
        # 订阅第一个控制话题
        control_topic = self.get_parameter('control_topic').value
        self.control_subscription = self.create_subscription(
            String,
            control_topic,
            self.control_callback,
            10  # QoS depth = 10，确保信号不丢失
        )
        self.get_logger().info(f'🎮 已订阅控制信号话题 1: {control_topic} (String类型，触发词: "操场")')
        
        # 订阅第二个控制话题（如果配置了且与话题1不同）
        control_topic_2 = self.get_parameter('control_topic_2').value
        if control_topic_2 and control_topic_2 != control_topic:
            self.control_subscription_2 = self.create_subscription(
                Int8,
                control_topic_2,
                self.control_callback_2,
                10  # QoS depth = 10，确保信号不丢失
            )
            self.get_logger().info(f'🎮 已订阅控制信号话题 2: {control_topic_2} (Int8类型，期望值: 1=caption, 2=text_cap，触发发送)')
        else:
            self.control_subscription_2 = None
            if control_topic_2 == control_topic:
                self.get_logger().warn(f'⚠️  控制话题 2 与话题 1 相同，跳过重复订阅')
        
        # 创建结果发布者
        result_topic = self.get_parameter('result_topic').value
        self.caption_publisher = self.create_publisher(
            String,
            result_topic,
            10
        )
        self.get_logger().info(f'📤 已创建结果发布话题: {result_topic}')
        
        self.get_logger().info('✅ Qwen3-VL Control Node 初始化完成（模型未加载）')
        self.get_logger().info('⏳ 等待控制信号...')
    
    def image_callback(self, msg: ROSImage):
        """
        图像话题回调函数 - 持续接收，保存最新一帧（ROS2 模式）
        """
        with self.latest_image_lock:
            self.latest_image_msg = msg
    
    def _start_rtsp_stream(self, rtsp_url: str):
        """
        启动 RTSP 流读取线程
        
        Args:
            rtsp_url: RTSP 流地址
        """
        self.get_logger().info(f'🔄 正在连接 RTSP 流: {rtsp_url}')
        
        # 创建 VideoCapture 对象
        self.rtsp_cap = cv2.VideoCapture(rtsp_url)
        
        if not self.rtsp_cap.isOpened():
            self.get_logger().error(f'❌ 无法打开 RTSP 流: {rtsp_url}')
            raise RuntimeError(f'无法打开 RTSP 流: {rtsp_url}')
        
        # 设置缓冲区大小（减少延迟）
        self.rtsp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.rtsp_running = True
        
        # 启动读取线程
        self.rtsp_thread = threading.Thread(target=self._rtsp_read_loop, daemon=True)
        self.rtsp_thread.start()
        
        self.get_logger().info(f'✅ RTSP 流读取线程已启动: {rtsp_url}')
    
    def _rtsp_read_loop(self):
        """
        RTSP 流读取循环（在独立线程中运行）
        """
        while self.rtsp_running:
            ret, frame = self.rtsp_cap.read()
            if ret:
                # 转换为 RGB 格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.latest_image_lock:
                    self.latest_rtsp_frame = frame_rgb
            else:
                self.get_logger().warn('⚠️  RTSP 流读取失败，尝试重新连接...')
                # 尝试重新连接
                self.rtsp_cap.release()
                import time
                time.sleep(1)
                rtsp_url = self.get_parameter('rtsp_url').value
                self.rtsp_cap = cv2.VideoCapture(rtsp_url)
                if not self.rtsp_cap.isOpened():
                    self.get_logger().error(f'❌ RTSP 流重连失败: {rtsp_url}')
                    break
                self.rtsp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 清理资源
        if self.rtsp_cap is not None:
            self.rtsp_cap.release()
            self.get_logger().info('🔄 RTSP 流已关闭')
    
    def control_callback(self, msg: String):
        """
        控制信号回调函数 1
        msg.data: 当接收到 "操场" 时，开始加载模型进行解析，但不发送结果（只缓存）
        """
        trigger_word = msg.data.strip()
        
        if trigger_word != "操场":
            # 不是触发词，跳过处理
            self.get_logger().debug(f'收到控制信号: "{trigger_word}"，不是触发词 "操场"，跳过处理')
            return
        
        # 收到 "操场"，按需加载模型并处理（只缓存，不发送）
        # 使用默认的 caption_prompt 进行预解析
        caption_prompt = self.get_parameter('caption_prompt').value
        if not caption_prompt:
            self.get_logger().error('❌ caption_prompt 未配置，无法处理')
            return
        
        self.get_logger().info('收到控制信号 "操场": 开始处理图像（使用 caption_prompt 解析后缓存，等待 control_topic_2 发送）...')
        
        # 检查是否正在处理（避免重复处理）
        with self.processing_lock:
            if self.is_processing:
                self.get_logger().warn('上一次处理尚未完成，跳过本次请求')
                return
            self.is_processing = True
        
        try:
            # 获取最新图像（根据图像源类型）
            image_source = self.get_parameter('image_source').value
            
            if image_source == 'ros2':
                # ROS2 模式：从话题获取图像
                with self.latest_image_lock:
                    if self.latest_image_msg is None:
                        self.get_logger().warn('⚠️  尚未收到图像，无法处理')
                        return
                    image_msg = self.latest_image_msg
                self._process_with_model(image_msg, send_result=False, prompt_template=caption_prompt)
            elif image_source == 'rtsp':
                # RTSP 模式：从 RTSP 流获取图像
                with self.latest_image_lock:
                    if self.latest_rtsp_frame is None:
                        self.get_logger().warn('⚠️  尚未收到 RTSP 帧，无法处理')
                        return
                    frame = self.latest_rtsp_frame.copy()
                self._process_with_rtsp_frame(frame, send_result=False, prompt_template=caption_prompt)
            else:
                self.get_logger().error(f'❌ 不支持的图像源类型: {image_source}')
                return
            
        except Exception as e:
            self.get_logger().error(f'❌ 处理图像时出错: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            with self.processing_lock:
                self.is_processing = False
    
    def control_callback_2(self, msg: Int8):
        """
        控制信号回调函数 2（Int8 类型）
        msg.data: 当接收到 1 或 2 时：
        - 值为 1：使用 caption_prompt 进行解析
        - 值为 2：使用 text_cap_prompt 进行解析
        - 如果有缓存结果，直接发送缓存结果（不进行解析）
        - 如果没有缓存结果，开始加载模型进行解析并在解析完成后发送结果
        """
        signal = msg.data
        
        if signal not in [1, 2]:
            # 不是期望值，跳过处理
            self.get_logger().debug(f'收到控制信号: {signal}，不是期望值 1 或 2，跳过处理')
            return
        
        # 根据信号值选择 prompt
        if signal == 1:
            prompt_type = "caption"
            prompt_template = self.get_parameter('caption_prompt').value
            cached_result_var = 'cached_caption_result'
        else:  # signal == 2
            prompt_type = "text_cap"
            prompt_template = self.get_parameter('text_cap_prompt').value
            cached_result_var = 'cached_text_cap_result'
        
        if not prompt_template:
            self.get_logger().error(f'❌ {prompt_type}_prompt 未配置，无法处理')
            return
        
        # 检查是否有对应 prompt 的缓存结果
        with self.cache_lock:
            cached_result = getattr(self, cached_result_var)
            if cached_result is not None:
                # 有缓存结果，直接发送，不进行解析
                self.get_logger().info(f'📤 收到控制信号 {signal} ({prompt_type}): 检测到缓存结果，直接发送（跳过解析）')
                setattr(self, cached_result_var, None)  # 清空对应缓存，避免重复使用
                self._publish_caption(cached_result)
                print("\033[36m" + "─" * 80 + "\033[0m")
                return
        
        # 没有缓存结果，开始解析并在解析完成后发送
        self.get_logger().info(f'收到控制信号 {signal} ({prompt_type}): 开始处理图像（解析完成后立即发送）...')
        
        # 检查是否正在处理（避免重复处理）
        with self.processing_lock:
            if self.is_processing:
                self.get_logger().warn('上一次处理尚未完成，跳过本次请求')
                return
            self.is_processing = True
        
        try:
            # 获取最新图像（根据图像源类型）
            image_source = self.get_parameter('image_source').value
            
            if image_source == 'ros2':
                # ROS2 模式：从话题获取图像
                with self.latest_image_lock:
                    if self.latest_image_msg is None:
                        self.get_logger().warn('⚠️  尚未收到图像，无法处理')
                        return
                    image_msg = self.latest_image_msg
                self._process_with_model(image_msg, send_result=True, prompt_template=prompt_template)
            elif image_source == 'rtsp':
                # RTSP 模式：从 RTSP 流获取图像
                with self.latest_image_lock:
                    if self.latest_rtsp_frame is None:
                        self.get_logger().warn('⚠️  尚未收到 RTSP 帧，无法处理')
                        return
                    frame = self.latest_rtsp_frame.copy()
                self._process_with_rtsp_frame(frame, send_result=True, prompt_template=prompt_template)
            else:
                self.get_logger().error(f'❌ 不支持的图像源类型: {image_source}')
                return
            
        except Exception as e:
            self.get_logger().error(f'❌ 处理图像时出错: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            with self.processing_lock:
                self.is_processing = False
    
    def _process_with_model(self, image_msg: ROSImage, send_result: bool = True, prompt_template: str = None):
        """
        按需加载模型，处理 ROS2 图像消息，然后释放资源
        
        Args:
            image_msg: ROS2 Image 消息
            send_result: 是否在解析完成后立即发送结果（True=立即发送，False=只缓存）
            prompt_template: 提示词模板（如果为 None，则使用默认的 caption_prompt）
        """
        # 1. 转换图像
        self.get_logger().info('🔄 转换图像...')
        pil_image = self._ros_image_to_pil(image_msg)
        self._process_image(pil_image, send_result=send_result, prompt_template=prompt_template)
    
    def _process_with_rtsp_frame(self, frame: np.ndarray, send_result: bool = True, prompt_template: str = None):
        """
        按需加载模型，处理 RTSP 帧，然后释放资源
        
        Args:
            frame: RTSP 帧（RGB numpy array）
            send_result: 是否在解析完成后立即发送结果（True=立即发送，False=只缓存）
            prompt_template: 提示词模板（如果为 None，则使用默认的 caption_prompt）
        """
        # 1. 转换图像
        self.get_logger().info('🔄 转换图像...')
        pil_image = Image.fromarray(frame)
        self._process_image(pil_image, send_result=send_result, prompt_template=prompt_template)
    
    def _process_image(self, pil_image: Image.Image, send_result: bool = True, prompt_template: str = None):
        """
        处理图像（通用方法，支持 ROS2 和 RTSP）
        
        Args:
            pil_image: PIL Image 对象
            send_result: 是否在解析完成后立即发送结果（True=立即发送，False=只缓存）
            prompt_template: 提示词模板（如果为 None，则使用默认的 caption_prompt）
        """
        caption_generator = None
        try:
            # 1.1 根据 flip 参数决定是否翻转图像
            flip = self.get_parameter('flip').value
            if flip:
                self.get_logger().info('🔄 正在将图像旋转180度...')
                pil_image = pil_image.rotate(180)
            
            # 1.2 如果没有提供 prompt_template，使用默认的 caption_prompt
            if prompt_template is None:
                prompt_template = self.get_parameter('caption_prompt').value
                if not prompt_template:
                    self.get_logger().error('❌ caption_prompt 未配置，无法处理')
                    return
            
            # 1.3 确定使用的 prompt 类型（用于缓存）
            caption_prompt = self.get_parameter('caption_prompt').value
            text_cap_prompt = self.get_parameter('text_cap_prompt').value
            
            # 判断 prompt_template 属于哪种类型
            if prompt_template == caption_prompt:
                prompt_type = 'caption'
                cached_result_var = 'cached_caption_result'
            elif prompt_template == text_cap_prompt:
                prompt_type = 'text_cap'
                cached_result_var = 'cached_text_cap_result'
            else:
                # 未知的 prompt，默认使用 caption 缓存
                prompt_type = 'caption'
                cached_result_var = 'cached_caption_result'
                self.get_logger().warn(f'⚠️  未知的 prompt_template，使用 caption 缓存')
            
            # 2. 加载模型（按需加载）
            self.get_logger().info('🔄 正在加载 Qwen3-VL 模型（按需加载）...')
            caption_generator = Qwen3VLCaption(
                model_path=self.get_parameter('model_path').value,
                prompt_template=prompt_template,
                trust_remote_code=self.get_parameter('trust_remote_code').value,
                max_new_tokens=self.get_parameter('max_new_tokens').value,
                temperature=self.get_parameter('temperature').value,
                top_p=self.get_parameter('top_p').value,
                do_sample=self.get_parameter('do_sample').value,
            )
            
            # 3. 生成描述
            caption = caption_generator.generate_caption(pil_image)
            
            self.get_logger().info(f'✅ 生成描述: {caption}')
            
            # 4. 根据 send_result 参数决定是发送还是缓存
            with self.cache_lock:
                if send_result:
                    # 立即发送结果
                    self.get_logger().info('📤 解析完成，立即发送结果')
                    self._publish_caption(caption)
                    # 发送后清空对应缓存（确保不会重复使用）
                    setattr(self, cached_result_var, None)
                else:
                    # 只缓存结果，不发送（根据 prompt_type 缓存到对应的变量）
                    self.get_logger().info(f'⏳ 解析完成，缓存结果到 {prompt_type} 缓存，等待 control_topic_2 信号发送...')
                    setattr(self, cached_result_var, caption)
            
        finally:
            # 5. 释放模型资源
            if caption_generator is not None:
                self.get_logger().info('🔄 正在释放模型资源...')
                self._cleanup_model(caption_generator)
                self.get_logger().info('✅ 模型资源已释放')
                print("\033[36m" + "─" * 80 + "\033[0m")
    
    def _ros_image_to_pil(self, msg: ROSImage) -> Image.Image:
        """
        将 ROS2 sensor_msgs/Image 转换为 PIL.Image
        
        Args:
            msg: ROS2 Image 消息
            
        Returns:
            PIL.Image 对象（RGB 格式）
        """
        try:
            # 转换为 OpenCV 格式 (BGR)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 转换为 RGB
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # 转换为 PIL.Image
            pil_image = Image.fromarray(cv_image_rgb)
            return pil_image
        except Exception as e:
            self.get_logger().error(f'❌ 图像转换失败: {e}')
            raise
    
    def _publish_caption(self, caption: str):
        """
        发布描述结果
        
        Args:
            caption: 图像描述文本
        """
        msg = String()
        msg.data = caption
        self.caption_publisher.publish(msg)
        self.get_logger().debug(f'📤 已发布描述结果')
    
    def _cleanup_model(self, caption_generator: Qwen3VLCaption):
        """
        清理模型资源

        Args:
            caption_generator: Qwen3VLCaption 实例
        """
        try:
            # 删除模型和处理器
            if hasattr(caption_generator, 'model'):
                del caption_generator.model
            if hasattr(caption_generator, 'processor'):
                del caption_generator.processor
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            self.get_logger().warn(f'⚠️  清理资源时出错: {e}')
    
    def destroy_node(self):
        """
        节点销毁时清理资源
        """
        # 停止 RTSP 流读取
        if self.rtsp_running:
            self.rtsp_running = False
            if self.rtsp_thread is not None:
                self.rtsp_thread.join(timeout=2.0)
            if self.rtsp_cap is not None:
                self.rtsp_cap.release()
        
        super().destroy_node()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用 Qwen3-VL 模型生成图像描述（精简版，支持命令行和 ROS2 模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 命令行模式
  python semantic_ros2.py --image path/to/image.jpg --model_path "Qwen/Qwen3-VL-2B-Instruct" --task_type detailed_cap

  # ROS2 模式
  python semantic_ros2.py --ros2
        """,
    )

    parser.add_argument(
        "--ros2",
        action="store_true",
        help="以 ROS2 节点模式运行（需要 ROS2 环境）",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="输入图像路径（命令行模式必需）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Qwen3-VL 模型路径（本地路径或 HuggingFace 模型 ID，如 Qwen/Qwen3-VL-2B-Instruct）",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="",
        help="提示词模板（如果为空，将使用默认的 caption prompt）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="最大生成 token 数（默认: 1024）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度（默认: 0.7）",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Nucleus sampling 参数（默认: 0.8）",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="使用采样生成（默认: True）",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="是否信任远程代码（默认: True）",
    )

    # 使用 parse_known_args 以兼容 ROS2 的 --ros-args / --params-file 等参数
    args, _ = parser.parse_known_args()

    # ROS2 模式
    if args.ros2:
        try:
            rclpy.init()
            node = Qwen3VLControlNode()
            rclpy.spin(node)
        except KeyboardInterrupt:
            print("\n⚠️  收到中断信号，正在关闭节点...")
        except Exception as e:
            print(f"❌ ROS2 节点运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
        return

    # 命令行模式
    if not args.image:
        parser.error("命令行模式需要 --image 参数，或使用 --ros2 进入 ROS2 模式")

    try:
        # 创建模型实例
        # 如果没有提供 prompt_template，使用默认的 caption prompt
        if not args.prompt_template:
            args.prompt_template = "要求：1) 使用纯文本，不要使用markdown格式、分隔符、换行符等特殊字符；2) 不要使用'照片'、'图片'、'视角'、'画面'、'这张'等词汇，直接描述场景本身；3) 用一段连贯的文字描述，不要分段；4) 每个物体或特征只描述一次，不要重复描述相同的内容；5) 避免循环重复，描述要简洁完整"
        
        caption_generator = Qwen3VLCaption(
            model_path=args.model_path,
            prompt_template=args.prompt_template,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )

        # 生成描述
        caption = caption_generator.generate_caption(args.image)

        # 输出结果
        print("\n" + "=" * 60)
        print("📝 图像描述:")
        print("=" * 60)
        print(caption)
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

