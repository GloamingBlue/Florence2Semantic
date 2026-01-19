import os
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/ubun/robot/Florence2Semantic/models/qwen3-vl-2B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-2B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("/home/ubun/robot/Florence2Semantic/models/qwen3-vl-2B-Instruct")

# 展开 ~ 路径为绝对路径
image_path = os.path.expanduser("~/1.png")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "只识别显示器上的文字内容。要求：1)使用纯文本，不要使用markdown格式、分隔符等特殊字符"},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)


def number_to_cn(num: int) -> str:
    mapping = {
        0: "零",
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
        11: "十一",
        12: "十二",
        13: "十三",
        14: "十四",
        15: "十五",
        16: "十六",
        17: "十七",
        18: "十八",
        19: "十九",
        20: "二十",
        21: "二十一",
        22: "二十二",
        23: "二十三",
        24: "二十四",
        25: "二十五",
        26: "二十六",
        27: "二十七",
        28: "二十八",
        29: "二十九",
        30: "三十",
        31: "三十一",
    }
    return mapping.get(num, str(num))


def time_range_to_cn(time_range: str) -> str:
    normalized = time_range.replace("：", ":")
    if "-" not in normalized:
        return time_range
    start, end = normalized.split("-", 1)

    def format_time(part: str) -> str:
        if ":" not in part:
            return f"{number_to_cn(int(part))}点"
        hour_str, minute_str = part.split(":", 1)
        hour = number_to_cn(int(hour_str))
        minute = int(minute_str)
        if minute == 0:
            return f"{hour}点"
        return f"{hour}点{number_to_cn(minute)}分"

    return f"{format_time(start)}到{format_time(end)}"


def post_process_text(text: str) -> str:
    text = re.sub(r"[A-Za-z]+", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    title = lines[0]
    clauses = []
    tokens = [line for line in lines[1:] if not (line == "日期" or line == "考试科目" or line == "考试时间")]
    i = 0
    while i + 2 < len(tokens):
        date_token = tokens[i]
        subject = tokens[i + 1]
        time_token = tokens[i + 2]
        match = re.search(r"(\d{1,2})\.(\d{1,2})", date_token)
        if match:
            month = number_to_cn(int(match.group(1)))
            day = number_to_cn(int(match.group(2)))
            time_range = time_range_to_cn(time_token)
            clauses.append(f"{month}月{day}日考试科目为{subject}，考试时间{time_range}")
        i += 3
    if not clauses:
        return title
    return f"这是{title}，" + "，".join(clauses)


raw_text = output_text[0] if isinstance(output_text, list) and output_text else ""
print(raw_text)
print(post_process_text(raw_text))
