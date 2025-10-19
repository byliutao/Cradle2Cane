import os
import csv
import shutil
from tqdm import tqdm

def rename_images_by_skin_color(
    csv_path: str,
    src_img_dir: str,
    dst_img_dir: str
):
    os.makedirs(dst_img_dir, exist_ok=True)

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            original_filename = row["filename"]
            skin_color = row["raw_output"].lower()  # 处理空格为下划线
            if skin_color == "error" or skin_color == "unknown":
                continue  # 跳过出错或无法识别的项

            name, ext = os.path.splitext(original_filename)
            new_filename = f"{name}_{skin_color}{ext}"

            src_path = os.path.join(src_img_dir, original_filename)
            dst_path = os.path.join(dst_img_dir, new_filename)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: file not found - {src_path}")


# ✅ 执行
if __name__ == "__main__":
    csv_path = "others/cele_test_wo_bg/race_predictions.csv"              # 你生成的csv路径
    src_img_dir = "others/cele_test_wo_bg"                  # 原图文件夹
    dst_img_dir = "others/cele_test_wo_bg_race4_qwen"     # 保存新命名图像的文件夹

    rename_images_by_skin_color(csv_path, src_img_dir, dst_img_dir)
