#!/usr/bin/env python3
"""
Florence2 图像描述生成脚本
独立使用 Florence2 模型生成图像描述，不依赖 AnyLabeling GUI
"""

import warnings
import sys
import argparse
import time
from pathlib import Path
from unittest.mock import patch
from typing import Dict, Tuple, Union

warnings.filterwarnings("ignore")

try:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_imports
except ImportError as e:
    print(f"❌ 缺少必要的依赖包: {e}")
    print("请安装: pip install torch transformers pillow")
    sys.exit(1)

# 尝试导入 psutil 用于内存监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class Florence2Caption:
    """Florence2 图像描述生成器"""

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

        # 自动选择设备
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

        print(f"🔄 正在加载模型: {model_path}")
        print(f"📱 使用设备: {self.device}")
        print(f"🔢 数据类型: {self.torch_dtype}")

        # 测量模型加载时间
        load_start = time.perf_counter()

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
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )

        load_end = time.perf_counter()
        self.load_time = load_end - load_start
        print(f"✅ 模型加载完成 (耗时: {self.load_time:.2f} 秒)")

        # 初始化 CUDA events（如果使用 GPU）
        self.use_cuda_events = torch.cuda.is_available()
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            # 重置显存统计
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # 记录模型加载后的内存和显存占用
        self.initial_memory = self._get_memory_usage()
        self.initial_gpu_memory = self._get_gpu_memory_usage()

    def _get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况（MB）"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "rss": mem_info.rss / (1024 * 1024),  # MB
                "vms": mem_info.vms / (1024 * 1024),  # MB
            }
        return {"rss": 0.0, "vms": 0.0}

    def _get_gpu_memory_usage(self) -> Union[Dict[str, float], None]:
        """
        获取当前 GPU 显存使用情况（MB）
        
        返回的显存信息说明：
        - allocated: 已分配显存，PyTorch 实际用于存储张量数据的显存
        - reserved: 保留显存，PyTorch 从 CUDA 分配器保留的总显存
                   包括已分配的显存 + 缓存池中的显存（用于快速分配新张量）
        - max_allocated: 峰值已分配显存，自上次 reset_peak_memory_stats() 后的最大值
        
        注意：reserved >= allocated，因为 PyTorch 会保留一些显存作为缓存
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            return {
                "allocated": allocated,
                "reserved": reserved,
                "max_allocated": max_allocated,
            }
        return None

    def _format_bytes(self, bytes_value: float) -> str:
        """格式化字节数为可读格式"""
        if bytes_value < 1024:
            return f"{bytes_value:.2f} MB"
        elif bytes_value < 1024 * 1024:
            return f"{bytes_value / 1024:.2f} GB"
        else:
            return f"{bytes_value / (1024 * 1024):.2f} TB"

    def generate_caption(
        self, image_path: str, return_timing: bool = False
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        为图像生成描述

        Args:
            image_path: 图像文件路径
            return_timing: 是否返回时间统计信息

        Returns:
            如果 return_timing=False: 图像描述文本
            如果 return_timing=True: (图像描述文本, 时间统计字典)
        """
        timing_info = {}

        # 读取图像
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        read_start = time.perf_counter()
        print(f"📷 正在读取图像: {image_path}")
        image = Image.open(image_path).convert("RGB")
        read_end = time.perf_counter()
        timing_info["image_read"] = read_end - read_start

        # 预处理
        preprocess_start = time.perf_counter()
        prompt = self.task_token
        print(f"🔤 使用任务类型: {self.task_type} ({self.task_token})")

        # 记录推理前的内存和显存
        memory_before = self._get_memory_usage()
        gpu_memory_before = self._get_gpu_memory_usage()
        
        # 重置峰值显存统计
        if self.use_cuda_events:
            torch.cuda.reset_peak_memory_stats()

        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
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

        # 同步 GPU（如果使用）
        if self.use_cuda_events:
            torch.cuda.synchronize()

        preprocess_end = time.perf_counter()
        timing_info["preprocess"] = preprocess_end - preprocess_start

        # 生成描述
        print("🤖 正在生成描述...")
        
        # 使用 CUDA events 测量 GPU 推理时间（更准确）
        if self.use_cuda_events:
            self.start_event.record()
        else:
            inference_start = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
            )

        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()
            timing_info["inference"] = (
                self.start_event.elapsed_time(self.end_event) / 1000.0
            )  # 转换为秒
        else:
            inference_end = time.perf_counter()
            timing_info["inference"] = inference_end - inference_start

        # 统计生成的 token 数量
        num_generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
        timing_info["generated_tokens"] = num_generated_tokens
        if timing_info["inference"] > 0:
            timing_info["tokens_per_second"] = (
                num_generated_tokens / timing_info["inference"]
            )

        # 解码生成的文本
        decode_start = time.perf_counter()
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        decode_end = time.perf_counter()
        timing_info["decode"] = decode_end - decode_start

        # 后处理获取描述
        postprocess_start = time.perf_counter()
        results = self.processor.post_process_generation(
            generated_text, task=self.task_token, image_size=image.size
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

        postprocess_end = time.perf_counter()
        timing_info["postprocess"] = postprocess_end - postprocess_start

        # 记录推理后的内存和显存
        memory_after = self._get_memory_usage()
        gpu_memory_after = self._get_gpu_memory_usage()

        # 计算内存和显存使用
        timing_info["memory"] = {
            "before": memory_before,
            "after": memory_after,
            "delta": {
                "rss": memory_after["rss"] - memory_before["rss"],
                "vms": memory_after["vms"] - memory_before["vms"],
            },
            "peak_rss": memory_after["rss"] - self.initial_memory["rss"],
        }

        if gpu_memory_before and gpu_memory_after:
            timing_info["gpu_memory"] = {
                "before": gpu_memory_before,
                "after": gpu_memory_after,
                "delta": {
                    "allocated": gpu_memory_after["allocated"]
                    - gpu_memory_before["allocated"],
                    "reserved": gpu_memory_after["reserved"]
                    - gpu_memory_before["reserved"],
                },
                "peak_allocated": gpu_memory_after["max_allocated"]
                - self.initial_gpu_memory["allocated"]
                if self.initial_gpu_memory
                else gpu_memory_after["max_allocated"],
            }
        else:
            timing_info["gpu_memory"] = None

        # 计算总时间
        timing_info["total"] = sum(
            [
                timing_info["image_read"],
                timing_info["preprocess"],
                timing_info["inference"],
                timing_info["decode"],
                timing_info["postprocess"],
            ]
        )

        if return_timing:
            return final_caption, timing_info
        else:
            return final_caption

    def __del__(self):
        """清理资源"""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用 Florence2 模型生成图像描述",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python florence2_caption.py --image path/to/image.jpg --model_path microsoft/Florence-2-large-ft

  # 使用详细描述模式
  python florence2_caption.py --image path/to/image.jpg --model_path /path/to/model --task_type detailed_cap

  # 使用本地模型路径
  python florence2_caption.py --image path/to/image.jpg --model_path /home/user/models/florence
        """,
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图像路径",
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
        default="caption",
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
        "--show_timing",
        action="store_true",
        help="显示详细的推理时间统计",
    )

    args = parser.parse_args()

    try:
        # 创建模型实例
        model_init_start = time.perf_counter()
        caption_generator = Florence2Caption(
            model_path=args.model_path,
            task_type=args.task_type,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
        )
        model_init_end = time.perf_counter()
        model_init_time = model_init_end - model_init_start

        # 生成描述
        if args.show_timing:
            caption, timing_info = caption_generator.generate_caption(
                args.image, return_timing=True
            )
        else:
            caption = caption_generator.generate_caption(args.image)
            timing_info = None

        # 输出结果
        print("\n" + "=" * 60)
        print("📝 图像描述:")
        print("=" * 60)
        print(caption)
        print("=" * 60)

        # 显示时间统计
        if args.show_timing and timing_info:
            print("\n" + "=" * 60)
            print("⏱️  推理时间统计")
            print("=" * 60)
            print(f"模型加载时间:     {caption_generator.load_time:>8.3f} 秒")
            
            # 显示模型加载后的内存占用
            if caption_generator.initial_memory:
                print(f"模型内存占用:     {caption_generator.initial_memory['rss']:>8.2f} MB")
            if caption_generator.initial_gpu_memory:
                print(f"模型显存占用:     {caption_generator.initial_gpu_memory['allocated']:>8.2f} MB")
            print("-" * 60)
            print(f"图像读取时间:     {timing_info['image_read']:>8.3f} 秒")
            print(f"预处理时间:       {timing_info['preprocess']:>8.3f} 秒")
            print(f"模型推理时间:     {timing_info['inference']:>8.3f} 秒")
            print(f"文本解码时间:     {timing_info['decode']:>8.3f} 秒")
            print(f"后处理时间:       {timing_info['postprocess']:>8.3f} 秒")
            print("-" * 60)
            print(f"单次推理总时间:   {timing_info['total']:>8.3f} 秒")
            print(f"推理速度:         {1.0/timing_info['total']:>8.2f} FPS")
            if timing_info['inference'] > 0:
                print(f"生成 token 数:    {timing_info.get('generated_tokens', 0):>8d} tokens")
                print(f"生成速度:         {timing_info.get('tokens_per_second', 0):>8.2f} tokens/s")
            
            # 显示内存使用
            print("\n" + "-" * 60)
            print("💾 内存使用统计")
            print("-" * 60)
            if timing_info.get('memory'):
                mem = timing_info['memory']
                print(f"推理前内存:       {mem['before']['rss']:>8.2f} MB")
                print(f"推理后内存:       {mem['after']['rss']:>8.2f} MB")
                print(f"内存增量:         {mem['delta']['rss']:>8.2f} MB")
                print(f"峰值内存增量:     {mem['peak_rss']:>8.2f} MB")
            
            # 显示显存使用
            if timing_info.get('gpu_memory'):
                print("\n" + "-" * 60)
                print("🎮 显存使用统计")
                print("-" * 60)
                gpu_mem = timing_info['gpu_memory']
                print(f"推理前已分配显存: {gpu_mem['before']['allocated']:>8.2f} MB")
                print(f"推理后已分配显存: {gpu_mem['after']['allocated']:>8.2f} MB")
                print(f"显存增量:         {gpu_mem['delta']['allocated']:>8.2f} MB")
                print(f"峰值显存增量:     {gpu_mem['peak_allocated']:>8.2f} MB")
                print(f"保留显存:         {gpu_mem['after']['reserved']:>8.2f} MB (已分配 + 缓存池)")
                print(f"缓存池大小:       {gpu_mem['after']['reserved'] - gpu_mem['after']['allocated']:>8.2f} MB")
            
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

