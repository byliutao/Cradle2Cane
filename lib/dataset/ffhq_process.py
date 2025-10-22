import json
import os
import numpy as np
from PIL import Image, ImageDraw
from lib.model.carvekit import CarveKit
import torch
from tqdm import tqdm
from lib.utils import qwen_utils

def format_int_to_filename(num):
    return f"{num:05}.png"


def expand_to_square(x_min, y_min, x_max, y_max, image_size, expand_ratio=0.2, offset_y=0):
    """
    计算并扩展一个正方形裁剪区域，并向上平移 offset_y 像素
    """
    width = x_max - x_min
    height = y_max - y_min

    # 计算最大边长，使其变成正方形
    max_side = max(width, height)
    expand_size = max_side * expand_ratio

    # 计算中心点
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2 - offset_y  # 向上平移 offset_y 像素

    # 计算新的正方形坐标
    new_x_min = max(0, int(center_x - max_side / 2 - expand_size))
    new_x_max = min(image_size[0], int(center_x + max_side / 2 + expand_size))

    new_y_min = max(0, int(center_y - max_side / 2 - expand_size))
    new_y_max = min(image_size[1], int(center_y + max_side / 2 + expand_size))

    return new_x_min, new_y_min, new_x_max, new_y_max


def apply_background(image, background_type="transparent"):
    """
    在 CarveKit 处理后，重新填充背景：
    - "transparent": 透明
    - "black": 纯黑色
    - "noise": 随机噪声
    """
    image = image.convert("RGBA")  # 确保有 Alpha 通道
    alpha = image.split()[-1]  # 获取 Alpha 通道

    if background_type == "transparent":
        return image  # 直接返回透明背景

    elif background_type == "black":
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))  # 纯黑背景
        return Image.composite(image, background, alpha)

    elif background_type == "noise":
        noise_array = np.random.randint(0, 256, (image.size[1], image.size[0], 3), dtype=np.uint8)
        background = Image.fromarray(noise_array).convert("RGBA")
        return Image.composite(image, background, alpha)

    else:
        raise ValueError("background_type 只能是 'transparent'、'black' 或 'noise'")

# 读取 JSON 文件
image_dir = "/home/u2120240694/data/dataset/ffhq512/"
json_path = "/home/u2120240694/data/dataset/ffhq-dataset-v2.json"
save_dir = "/home/u2120240694/data/dataset/ffhq_crop_gray_wo_crop"
info_json_path = "/home/u2120240694/data/dataset/json"


crop = False
expand_ratio = 0.6  
offset_y = 200  # **让正方形向上移动 50 像素**
background_type = "transparent"  # 选择 "transparent", "black", "noise"

os.makedirs(save_dir, exist_ok=True)

# CarveKit 初始化
# carvekit = CarveKit("/data/model/carvekit/tracer_b7.pth", "/data/model/carvekit/fba_matting.pth", "cuda", torch.float32)
carvekit = CarveKit("/home/u2120240694/data/model/tracer_b7.pth", "/home/u2120240694/data/model/fba_matting.pth", "cuda", torch.float32)


with open(json_path, 'r') as json_file:
    data = json.load(json_file)


for idx in tqdm(range(0, 70000)):

    first_image_info = data.get(str(idx), {})  
    landmark = first_image_info["image"]["face_landmarks"]

    # **缩放 landmark 坐标**
    landmark_resize = np.array([[item[0] * 0.5, item[1] * 0.5] for item in landmark])

    # **计算正方形裁剪框**
    x_min = min(point[0] for point in landmark_resize)
    y_min = min(point[1] for point in landmark_resize)
    x_max = max(point[0] for point in landmark_resize)
    y_max = max(point[1] for point in landmark_resize)

    image_name = format_int_to_filename(idx)

    # **保存**
    json_name = image_name.replace(".png",".json")
    info_json_path_single = os.path.join(info_json_path, json_name)
    
    with open(info_json_path_single, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if len(config) == 1:
        age = config[0]["faceAttributes"]["age"]
        gender = config[0]["faceAttributes"]["gender"]
    else:
        continue

    output_path = os.path.join(save_dir, f"{age}_{gender}_{image_name}")

    if os.path.exists(output_path):
        continue

    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path).convert("RGB")

    if crop:
        # **获取正方形裁剪区域**
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = expand_to_square(x_min, y_min, x_max, y_max, image.size, expand_ratio, offset_y)

        # **裁剪图像**
        cropped_image = image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

        # **CarveKit 处理**
        processed_image = carvekit([cropped_image])[0]
    else:
        processed_image = carvekit([image])[0]

    # **统一背景**
    final_image = apply_background(processed_image, background_type).convert("RGB")

    if background_type == "transparent":
        final_image.save(output_path, "PNG")  # 透明背景必须保存 PNG
    else:
        final_image.save(output_path)  # 其他格式正常保存

    # break