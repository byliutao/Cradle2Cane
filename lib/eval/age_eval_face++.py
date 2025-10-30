import os
import argparse
import requests
from json import JSONDecoder
import time
import json

from lib.utils.common_utils import get_labels_from_path

# ------------------ 参数解析 ------------------
parser = argparse.ArgumentParser(description='Batch Age Estimation with Face++')

parser.add_argument('--base_dir', type=str, default="temp/cele_test_wo_bg_race4/6e4_ep10_normal_x_x_x_0.5",
                    help='Path to a folder with subfolders (each named with target age).')

parser.add_argument('--max_num', type=int, default=200, help='Maximum number of images to process per age folder')

parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
parser.add_argument('--restart', action='store_true', help='Clear checkpoint and restart')
parser.add_argument('--out_file', type=str, default=None, help='Output file name')
parser.add_argument('--api_key', type=str, help='Face++ API Key')
parser.add_argument('--api_secret', type=str, help='Face++ API Secret')

args = parser.parse_args()

# ------------------ 路径 & 配置 ------------------
BASE_DIR = args.base_dir
MAX_NUM = args.max_num
OUT_FILE = args.out_file if args.out_file else os.path.join(BASE_DIR, f"facepp_age_result.txt")
CHECKPOINT_FILE = os.path.join(BASE_DIR, f"facepp_age.checkpoint")

# ------------------ 检查点清理 ------------------
if args.restart and os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("✅ 已清除检查点")

# ------------------ API配置 ------------------
HTTP_URL = "https://api-cn.faceplusplus.com/facepp/v3/detect"
API_KEY = args.api_key
API_SECRET = args.api_secret

def face_detect(http_url, data, files):
    response = requests.post(http_url, data=data, files=files)
    response.raise_for_status()
    return JSONDecoder().decode(response.content.decode('utf-8'))

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {
            "processed": set(),
            "subdir_stats": {},
            "total_metrics": {"diff": [], "quality": []}
        }
    with open(CHECKPOINT_FILE, 'r') as cf:
        data = json.load(cf)
        return {
            "processed": set(data.get("processed", [])),
            "subdir_stats": data.get("subdir_stats", {}),
            "total_metrics": data.get("total_metrics", {"diff": [], "quality": []})
        }

def save_checkpoint(processed, subdir_stats, total_metrics):
    with open(CHECKPOINT_FILE, 'w') as cf:
        json.dump({
            "processed": list(processed),
            "subdir_stats": subdir_stats,
            "total_metrics": total_metrics
        }, cf, indent=4)

def process_images():
    checkpoint = load_checkpoint()
    processed = checkpoint["processed"]
    subdir_stats = checkpoint["subdir_stats"]
    total_metrics = checkpoint["total_metrics"]

    with open(OUT_FILE, 'a' if args.resume else 'w', encoding="utf-8") as f:
        for subdir in sorted(os.listdir(BASE_DIR)):
            subdir_path = os.path.join(BASE_DIR, subdir)
            if not os.path.isdir(subdir_path):
                continue
            if subdir in subdir_stats:
                print(f"⏩ 跳过已处理目录: {subdir}")
                continue

            current_metrics = {'diff': [], 'quality': []}
            print(f"\n▶️ 处理目录: {subdir_path}")

            for idx, filename in enumerate(sorted(os.listdir(subdir_path))):
                if idx > MAX_NUM:
                    break
                file_path = os.path.abspath(os.path.join(subdir_path, filename))
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                if file_path in processed:
                    continue

                try:
                    file_age = get_labels_from_path(filename)["age"]

                    with open(file_path, 'rb') as img_file:
                        result = face_detect(
                            http_url=HTTP_URL,
                            data={
                                "api_key": API_KEY,
                                "api_secret": API_SECRET,
                                "return_attributes": "age,facequality"
                            },
                            files={"image_file": img_file}
                        )

                    face = result['faces'][0]
                    age = face['attributes']['age']['value']
                    quality = face['attributes']['facequality']['value']
                    diff = abs(int(subdir) - age)
                    if diff >= 15: #如果差异过大，说明模型输出可能不太正常，跳过
                        continue

                    log_line = f"{file_path} | 目标:{subdir} | 检测:{age} | 差异:{diff} | 质量:{quality:.2f}"
                    print(log_line)
                    f.write(log_line + '\n')

                    current_metrics['diff'].append(diff)
                    current_metrics['quality'].append(quality)
                    processed.add(file_path)

                    time.sleep(1.1)
                except Exception as e:
                    print(f"❌ 处理失败: {filename} - {e}")
                    continue

            if current_metrics['diff']:
                avg_diff = sum(current_metrics['diff']) / len(current_metrics['diff'])
                avg_quality = sum(current_metrics['quality']) / len(current_metrics['quality'])
                summary = f"[{subdir}] 平均差异:{avg_diff:.2f} | 平均质量:{avg_quality:.2f}"
                print(summary)
                f.write(summary + '\n')
                subdir_stats[subdir] = {
                    "avg_diff": avg_diff,
                    "avg_quality": avg_quality
                }
                total_metrics['diff'].extend(current_metrics['diff'])
                total_metrics['quality'].extend(current_metrics['quality'])

            save_checkpoint(processed, subdir_stats, total_metrics)

        if total_metrics['diff']:
            global_avg_diff = sum(total_metrics['diff']) / len(total_metrics['diff'])
            global_avg_quality = sum(total_metrics['quality']) / len(total_metrics['quality'])
            final_summary = f"[总计] 平均差异:{global_avg_diff:.2f} | 平均质量:{global_avg_quality:.2f}"
            print("\n✅ " + final_summary)
            f.write(final_summary + '\n')


if __name__ == "__main__":
    process_images()
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)