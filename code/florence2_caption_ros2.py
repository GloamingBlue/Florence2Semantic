#!/usr/bin/env python3
"""
Florence2 图像描述生成脚本（精简版）
独立使用 Florence2 模型生成图像描述，不依赖 AnyLabeling GUI
精简版：移除了性能监测和时间计算功能，只保留核心语义生成功能
"""

import warnings
import sys
import argparse
import threading
import gc
import os
from pathlib import Path
from unittest.mock import patch
from typing import Union, Optional

warnings.filterwarnings("ignore")

# 翻译相关导入
from transformers import MarianMTModel, MarianTokenizer

try:
    import torch
    from PIL import Image
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoProcessor
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


class Florence2Caption:
    """Florence2 图像描述生成器（精简版）"""

    # 任务类型映射
    TASK_MAPPING = {
        "caption": "<CAPTION>",
        "detailed_cap": "<DETAILED_CAPTION>",
        "more_detailed_cap": "<MORE_DETAILED_CAPTION>",
    }

    def __init__(
        self,
        model_path: str,
        task_type: str = "caption",
        trust_remote_code: bool = True,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        num_beams: int = 3,
    ):
        """
        初始化 Florence2 模型

        Args:
            model_path: 模型路径（本地路径或 HuggingFace 模型 ID）
            task_type: 任务类型，可选 "caption", "detailed_cap", "more_detailed_cap"
            trust_remote_code: 是否信任远程代码
            max_new_tokens: 最大生成 token 数
            do_sample: 是否使用采样
            num_beams: beam search 的 beam 数量
        """
        if task_type not in self.TASK_MAPPING:
            raise ValueError(
                f"不支持的任务类型: {task_type}。"
                f"支持的类型: {list(self.TASK_MAPPING.keys())}"
            )

        self.task_type = task_type
        self.task_token = self.TASK_MAPPING[task_type]
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.num_beams = num_beams

        # 展开路径中的 ~ 符号为绝对路径
        # HuggingFace 库无法识别包含 ~ 的路径，需要展开
        if '~' in model_path:
            model_path = os.path.expanduser(model_path)
            # 展开后如果是相对路径，转换为绝对路径
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)

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
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=trust_remote_code,
                attn_implementation="eager",
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )

        print(f"✅ Caption 模型加载完成")
        
        # 翻译模型（可选，按需加载）
        self.translator = None
        self.translate_to_chinese = False

    def set_translation(
        self, 
        enable: bool = True, 
        model_name: str = "Helsinki-NLP/opus-mt-en-zh",
        model_path: Optional[str] = None
    ):
        """
        设置是否启用翻译功能
        
        Args:
            enable: 是否启用翻译
            model_name: 翻译模型名称（HuggingFace ID），当 model_path 为 None 时使用
                      - "Helsinki-NLP/opus-mt-en-zh" (推荐，英文到中文)
                      - "facebook/nllb-200-distilled-600M" (多语言，需要指定语言代码)
            model_path: 本地翻译模型路径（如果提供，将优先使用本地路径）
        """
        self.translate_to_chinese = enable
        
        if enable and self.translator is None:
            # 优先使用本地路径
            if model_path:
                # 展开路径中的 ~ 符号为绝对路径
                if '~' in model_path:
                    model_path = os.path.expanduser(model_path)
                    # 展开后如果是相对路径，转换为绝对路径
                    if not os.path.isabs(model_path):
                        model_path = os.path.abspath(model_path)
                
                if Path(model_path).exists():
                    print(f"🔄 正在从本地路径加载翻译模型: {model_path}")
                    try:
                        self.translator_tokenizer = MarianTokenizer.from_pretrained(model_path)
                        self.translator_model = MarianMTModel.from_pretrained(model_path)
                        if torch.cuda.is_available():
                            self.translator_model = self.translator_model.to(self.device)
                        print(f"✅ 翻译模型加载完成（本地路径）")
                    except Exception as e:
                        print(f"⚠️  从本地路径加载翻译模型失败: {e}")
                        self.translate_to_chinese = False
                else:
                    print(f"⚠️  本地翻译模型路径不存在: {model_path}，将使用 HuggingFace 模型")
                    # 回退到 HuggingFace 模型
                    model_path = None
            
            if enable and model_path is None:
                print(f"🔄 正在从 HuggingFace 加载翻译模型: {model_name}")
                try:
                    self.translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
                    self.translator_model = MarianMTModel.from_pretrained(model_name)
                    if torch.cuda.is_available():
                        self.translator_model = self.translator_model.to(self.device)
                    print(f"✅ 翻译模型加载完成（HuggingFace）")
                except Exception as e:
                    print(f"⚠️  翻译模型加载失败: {e}")
                    self.translate_to_chinese = False

    def _translate_to_chinese(self, text: str) -> str:
        """
        将英文文本翻译为中文
        
        Args:
            text: 英文文本
            
        Returns:
            中文文本
        """
        if not self.translate_to_chinese or self.translator_model is None:
            return text
        
        try:
            # 翻译
            inputs = self.translator_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = self.translator_model.generate(**inputs, max_length=512)
            
            translated_text = self.translator_tokenizer.decode(translated[0], skip_special_tokens=True)
            return translated_text
        except Exception as e:
            print(f"⚠️  翻译失败: {e}，返回原文")
            return text

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

        # 预处理
        prompt = self.task_token
        inputs = self.processor(
            text=prompt, images=pil_image, return_tensors="pt"
        )

        # 将输入移动到设备并匹配模型数据类型
        model_dtype = next(self.model.parameters()).dtype
        inputs = {
            k: (
                v.to(device=self.device, dtype=model_dtype)
                if torch.is_floating_point(v)
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }

        # 生成描述
        print("🤖 正在生成描述...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                use_cache=False,  # 禁用缓存以避免 past_key_values 为 None 的问题
            )

        # 解码生成的文本
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # 后处理获取描述
        results = self.processor.post_process_generation(
            generated_text, task=self.task_token, image_size=pil_image.size
        )

        # 提取描述文本
        if self.task_token in results:
            caption = results[self.task_token]
            if isinstance(caption, str):
                final_caption = caption
            elif isinstance(caption, dict) and "caption" in caption:
                final_caption = caption["caption"]
            else:
                final_caption = str(caption)
        else:
            final_caption = generated_text

        # 如果启用了翻译，将英文翻译为中文
        if self.translate_to_chinese:
            print("🔄 正在翻译为中文...")
            final_caption = self._translate_to_chinese(final_caption)

        return final_caption

    def __del__(self):
        """清理资源"""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "translator_model"):
            del self.translator_model
        if hasattr(self, "translator_tokenizer"):
            del self.translator_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ROS2 轻量级控制节点（不加载模型）
class Florence2ControlNode(Node):
    """
    轻量级控制节点，负责：
    - 持续接收图像流，保存最新一帧
    - 监听控制信号
    - 收到信号 1 时，按需加载模型、处理图像、释放资源
    """
    
    def __init__(self):
        super().__init__('florence2_control_node')
        
        # 参数声明
        self.declare_parameter('image_source', 'ros2')  # 图像来源: "ros2" 或 "rtsp"
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')  # ROS2 图像话题
        self.declare_parameter('rtsp_url', 'rtsp://192.168.168.168:8554/test')  # RTSP 流地址
        self.declare_parameter('control_topic', '/navigation/florence')  # 控制信号话题 1 (String类型，触发词: "操场")
        self.declare_parameter('control_topic_2', '/nav/arrival')  # 控制信号话题 2 (Int8类型，期望值: 1，触发发送)
        self.declare_parameter('model_path', '/home/ubun/xanylabeling_data/models/florence')
        self.declare_parameter('task_type', 'more_detailed_cap')
        self.declare_parameter('result_topic', '/florence2/caption')
        self.declare_parameter('max_new_tokens', 1024)
        self.declare_parameter('num_beams', 3)
        self.declare_parameter('do_sample', False)
        self.declare_parameter('trust_remote_code', True)
        self.declare_parameter('translate_to_chinese', True)  # 是否翻译为中文
        self.declare_parameter('translation_model', 'Helsinki-NLP/opus-mt-en-zh')  # 翻译模型（HuggingFace ID）
        self.declare_parameter('translation_model_path', '')  # 翻译模型本地路径（可选）
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
        
        # 结果缓存
        self.cached_result = None
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
            self.get_logger().info(f'🎮 已订阅控制信号话题 2: {control_topic_2} (Int8类型，期望值: 1，触发发送)')
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
        
        self.get_logger().info('✅ Florence2 Control Node 初始化完成（模型未加载）')
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
        self.get_logger().info('收到控制信号 "操场": 开始处理图像（解析后缓存，等待 control_topic_2 发送）...')
        
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
                self._process_with_model(image_msg, send_result=False)
            elif image_source == 'rtsp':
                # RTSP 模式：从 RTSP 流获取图像
                with self.latest_image_lock:
                    if self.latest_rtsp_frame is None:
                        self.get_logger().warn('⚠️  尚未收到 RTSP 帧，无法处理')
                        return
                    frame = self.latest_rtsp_frame.copy()
                self._process_with_rtsp_frame(frame, send_result=False)
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
        msg.data: 当接收到 1 时：
        - 如果有缓存结果，直接发送缓存结果（不进行解析）
        - 如果没有缓存结果，开始加载模型进行解析并在解析完成后发送结果
        """
        signal = msg.data
        
        if signal != 1:
            # 不是期望值，跳过处理
            self.get_logger().debug(f'收到控制信号: {signal}，不是期望值 1，跳过处理')
            return
        
        # 检查是否有缓存结果
        with self.cache_lock:
            if self.cached_result is not None:
                # 有缓存结果，直接发送，不进行解析
                self.get_logger().info('📤 收到控制信号 1: 检测到缓存结果，直接发送（跳过解析）')
                cached = self.cached_result
                self.cached_result = None  # 清空缓存，避免重复使用
                self._publish_caption(cached)
                print("\033[36m" + "─" * 80 + "\033[0m")
                return
        
        # 没有缓存结果，开始解析并在解析完成后发送
        self.get_logger().info('收到控制信号 1: 开始处理图像（解析完成后立即发送）...')
        
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
                self._process_with_model(image_msg, send_result=True)
            elif image_source == 'rtsp':
                # RTSP 模式：从 RTSP 流获取图像
                with self.latest_image_lock:
                    if self.latest_rtsp_frame is None:
                        self.get_logger().warn('⚠️  尚未收到 RTSP 帧，无法处理')
                        return
                    frame = self.latest_rtsp_frame.copy()
                self._process_with_rtsp_frame(frame, send_result=True)
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
    
    def _process_with_model(self, image_msg: ROSImage, send_result: bool = True):
        """
        按需加载模型，处理 ROS2 图像消息，然后释放资源
        
        Args:
            image_msg: ROS2 Image 消息
            send_result: 是否在解析完成后立即发送结果（True=立即发送，False=只缓存）
        """
        # 1. 转换图像
        self.get_logger().info('🔄 转换图像...')
        pil_image = self._ros_image_to_pil(image_msg)
        self._process_image(pil_image, send_result=send_result)
    
    def _process_with_rtsp_frame(self, frame: np.ndarray, send_result: bool = True):
        """
        按需加载模型，处理 RTSP 帧，然后释放资源
        
        Args:
            frame: RTSP 帧（RGB numpy array）
            send_result: 是否在解析完成后立即发送结果（True=立即发送，False=只缓存）
        """
        # 1. 转换图像
        self.get_logger().info('🔄 转换图像...')
        pil_image = Image.fromarray(frame)
        self._process_image(pil_image, send_result=send_result)
    
    def _process_image(self, pil_image: Image.Image, send_result: bool = True):
        """
        处理图像（通用方法，支持 ROS2 和 RTSP）
        
        Args:
            pil_image: PIL Image 对象
            send_result: 是否在解析完成后立即发送结果（True=立即发送，False=只缓存）
        """
        caption_generator = None
        try:
            # 1.1 根据 flip 参数决定是否翻转图像
            flip = self.get_parameter('flip').value
            if flip:
                self.get_logger().info('🔄 正在将图像旋转180度...')
                pil_image = pil_image.rotate(180)
            
            # 2. 加载模型（按需加载）
            self.get_logger().info('🔄 正在加载 Florence2 模型（按需加载）...')
            caption_generator = Florence2Caption(
                model_path=self.get_parameter('model_path').value,
                task_type=self.get_parameter('task_type').value,
                max_new_tokens=self.get_parameter('max_new_tokens').value,
                num_beams=self.get_parameter('num_beams').value,
                do_sample=self.get_parameter('do_sample').value,
                trust_remote_code=self.get_parameter('trust_remote_code').value,
            )
            
            # 2.1 如果启用翻译，加载翻译模型
            translate_to_chinese = self.get_parameter('translate_to_chinese').value
            if translate_to_chinese:
                translation_model = self.get_parameter('translation_model').value
                translation_model_path = self.get_parameter('translation_model_path').value
                # 如果路径为空字符串，则使用 None（表示使用 HuggingFace 模型）
                if translation_model_path == '':
                    translation_model_path = None
                caption_generator.set_translation(
                    enable=True, 
                    model_name=translation_model,
                    model_path=translation_model_path
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
                    # 发送后清空缓存（确保不会重复使用）
                    self.cached_result = None
                else:
                    # 只缓存结果，不发送
                    self.get_logger().info('⏳ 解析完成，缓存结果，等待 control_topic_2 信号发送...')
                    self.cached_result = caption
            
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
    
    def _cleanup_model(self, caption_generator: Florence2Caption):
        """
        清理模型资源
        
        Args:
            caption_generator: Florence2Caption 实例
        """
        try:
            # 删除模型和处理器
            if hasattr(caption_generator, 'model'):
                del caption_generator.model
            if hasattr(caption_generator, 'processor'):
                del caption_generator.processor
            # 删除翻译模型（如果存在）
            if hasattr(caption_generator, 'translator_model'):
                del caption_generator.translator_model
            if hasattr(caption_generator, 'translator_tokenizer'):
                del caption_generator.translator_tokenizer
            
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
        description="使用 Florence2 模型生成图像描述（精简版，支持命令行和 ROS2 模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 命令行模式
  python florence2_caption_ros2_lite.py --image path/to/image.jpg --model_path /path/to/model --task_type detailed_cap

  # ROS2 模式
  python florence2_caption_ros2_lite.py --ros2
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
        default="/home/ubun/xanylabeling_data/models/florence",
        help="模型路径（本地路径或 HuggingFace 模型 ID，如 microsoft/Florence-2-large-ft）",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="more_detailed_cap",
        choices=["caption", "detailed_cap", "more_detailed_cap"],
        help="任务类型: caption (基础描述), detailed_cap (详细描述), more_detailed_cap (更详细描述)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="最大生成 token 数（默认: 1024）",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help="Beam search 的 beam 数量（默认: 3）",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="使用采样生成（默认: False，使用 beam search）",
    )
    parser.add_argument(
        "--translate_to_chinese",
        action="store_true",
        help="将生成的英文描述翻译为中文（需要额外加载翻译模型）",
    )
    parser.add_argument(
        "--translation_model",
        type=str,
        default="Helsinki-NLP/opus-mt-en-zh",
        help="翻译模型名称（HuggingFace ID，默认: Helsinki-NLP/opus-mt-en-zh）",
    )
    parser.add_argument(
        "--translation_model_path",
        type=str,
        default=None,
        help="翻译模型本地路径（如果提供，将优先使用本地路径而不是 HuggingFace 模型）",
    )

    # 使用 parse_known_args 以兼容 ROS2 的 --ros-args / --params-file 等参数
    args, _ = parser.parse_known_args()

    # ROS2 模式
    if args.ros2:
        try:
            rclpy.init()
            node = Florence2ControlNode()
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
        caption_generator = Florence2Caption(
            model_path=args.model_path,
            task_type=args.task_type,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
        )
        
        # 如果启用翻译，设置翻译功能
        if args.translate_to_chinese:
            caption_generator.set_translation(
                enable=True, 
                model_name=args.translation_model,
                model_path=args.translation_model_path
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

