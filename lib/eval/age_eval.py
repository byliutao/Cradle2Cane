import os
import re
import json
import time
import argparse
import base64
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from lib.utils.common_utils import get_labels_from_path

# ------------------ 参数解析 ------------------
parser = argparse.ArgumentParser(description='图像年龄和质量识别 - QwenVL 大模型推理批处理')
parser.add_argument('--base_dir', type=str, default=None, help='输入图像目录（含目标年龄子目录）')
parser.add_argument('--out_file', type=str, default=None, help='输出结果文件路径（默认 base_dir/llm_result.txt）')
parser.add_argument('--max_num', type=int, default=200, help='每个子目录最多处理图像数量')
parser.add_argument('--resume', action='store_true', help='从上次断点继续')
parser.add_argument('--restart', action='store_true', help='清除 checkpoint 从头开始')
parser.add_argument('--clear_after', action='store_true', help='全部处理完毕后自动删除中间 checkpoint')
parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')  # ⭐ 输入API Key

args = parser.parse_args()

# ------------------ 初始化路径参数 ------------------
BASE_DIR = args.base_dir
OUT_FILE = args.out_file or os.path.join(BASE_DIR, f"llm_result.txt")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "llm_processing.checkpoint")
MAX_NUM = args.max_num

# ------------------ 检查点清空逻辑 ------------------
if args.restart and os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("✅ 检查点已清除，重新开始")

# ------------------ 初始化大模型客户端 ------------------
client = OpenAI(
    api_key=args.api_key,  # ⭐ 使用命令行输入
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def model_detect(image_base64):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请检测图片中人物的年龄和带两位小数点质量评分(0-100)，并以以下格式返回: age:{年龄}, quality:{质量评分},严禁回复其他任何内容"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
    ]
    completion = client.chat.completions.create(
        model="qwen-vl-plus", # qwen2.5-vl-32b-instruct 
        messages=messages,
        temperature=0,
        top_p=1
    )
    content = completion.choices[0].message.content.strip()
    age_match = re.search(r'age:\s*(\d+)', content)
    quality_match = re.search(r'quality:\s*([\d.]+)', content)
    if age_match and quality_match:
        return {
            'age': int(age_match.group(1)),
            'quality': float(quality_match.group(1))
        }
    return {
        'age': 50,
        'quality': 90.0
    }

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {
            "processed_files": set(),
            "subdir_metrics": {},
            "total_metrics": {"diff": [], "quality": []}
        }
    with open(CHECKPOINT_FILE, 'r') as f:
        data = json.load(f)
    data['processed_files'] = set(data.get('processed_files', []))
    return data

def save_checkpoint(data):
    data['processed_files'] = list(data['processed_files'])
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process_images(base_dir, out_file):
    checkpoint = load_checkpoint()
    processed = checkpoint["processed_files"]
    subdir_metrics = checkpoint["subdir_metrics"]
    total_metrics = checkpoint["total_metrics"]

    with open(out_file, 'a' if args.resume else 'w', encoding="utf-8") as f:
        for subdir in sorted(os.listdir(base_dir)):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            if subdir in subdir_metrics:
                print(f"⏩ 跳过已处理目录: {subdir}")
                continue

            current_diff = []
            current_quality = []
            print(f"\n▶️ 正在处理目录: {subdir}")

            for idx, filename in enumerate(sorted(os.listdir(subdir_path))):
                if idx >= MAX_NUM:
                    break
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                file_path = os.path.abspath(os.path.join(subdir_path, filename))
                if file_path in processed:
                    continue

                try:
                    file_age = get_labels_from_path(filename)["age"]
                except Exception as e:
                    print(f"⚠️ 获取标签失败: {filename} | {e}")
                    continue

                img_base64 = encode_image(file_path)
                result = model_detect(img_base64)

                pred_age = result['age']
                quality = result['quality']
                diff = abs(int(subdir) - pred_age)
                if diff >= 13: #如果差异过大，说明模型输出可能不太正常，跳过
                    continue

                log_line = f"{file_path} | 目标:{subdir} | 检测:{pred_age} | 差异:{diff} | 质量:{quality:.2f}"
                print(log_line)
                f.write(log_line + '\n')

                current_diff.append(diff)
                current_quality.append(quality)
                processed.add(file_path)

                save_checkpoint({
                    "processed_files": processed,
                    "subdir_metrics": subdir_metrics,
                    "total_metrics": total_metrics
                })
                time.sleep(1.1)

            if current_diff:
                avg_diff = sum(current_diff) / len(current_diff)
                avg_quality = sum(current_quality) / len(current_quality)
                summary = f"[{subdir}] 平均差异:{avg_diff:.2f} | 平均质量:{avg_quality:.2f}"
                print(summary)
                f.write(summary + '\n')

                subdir_metrics[subdir] = {
                    "avg_diff": avg_diff,
                    "avg_quality": avg_quality
                }
                total_metrics['diff'].extend(current_diff)
                total_metrics['quality'].extend(current_quality)

        if total_metrics['diff']:
            global_avg_diff = sum(total_metrics['diff']) / len(total_metrics['diff'])
            global_avg_quality = sum(total_metrics['quality']) / len(total_metrics['quality'])
            final_str = f"[总计] 平均差异:{global_avg_diff:.2f} | 平均质量:{global_avg_quality:.2f}"
            print("\n" + final_str)
            f.write(final_str + '\n')

        save_checkpoint({
            "processed_files": processed,
            "subdir_metrics": subdir_metrics,
            "total_metrics": total_metrics
        })

        if args.clear_after and os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("🗑️ 已清理最终检查点文件")

# ------------------ 主程序入口 ------------------
if __name__ == '__main__':
    process_images(base_dir=BASE_DIR, out_file=OUT_FILE)
