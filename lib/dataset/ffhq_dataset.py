from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image
import yaml
import os
import random
import torch
import numpy as np


def resize_and_pad_image(img, output_size=(512, 512), pad_mode="gray"):
    """
    将输入图像填充为指定大小：
    - 若图像较大，进行中心裁剪；
    - 若图像较小，在四周填充黑色或随机噪声。

    参数：
    - image_path: 图像文件路径
    - output_size: 目标大小 (默认 512x512)
    - pad_mode: 填充模式 ("black" - 黑色填充, "noise" - 随机噪声填充)
    
    返回：
    - 处理后的 PIL Image 对象
    """

    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    
    if img.size == output_size:
        return img
    
    original_width, original_height = img.size
    target_width, target_height = output_size

    # 计算裁剪区域
    if original_width > target_width or original_height > target_height:
        left = max(0, (original_width - target_width) // 2)
        top = max(0, (original_height - target_height) // 2)
        right = left + target_width
        bottom = top + target_height
        img = img.crop((left, top, right, bottom))
    else:
        # 计算填充尺寸
        pad_left = (target_width - original_width) // 2
        pad_top = (target_height - original_height) // 2
        pad_right = target_width - original_width - pad_left
        pad_bottom = target_height - original_height - pad_top

        if pad_mode == "black":
            background = Image.new("RGB", output_size, (0, 0, 0))
        elif pad_mode == "gray":
            background = Image.new("RGB", output_size, (128, 128, 128))            
        elif pad_mode == "noise":
            noise = np.random.randint(0, 256, (target_height, target_width, 3), dtype=np.uint8)
            background = Image.fromarray(noise)
        else:
            raise ValueError("pad_mode 只能是 'black' 或 'noise'")

        # 将原图粘贴到中心
        background.paste(img, (pad_left, pad_top))
        img = background

    return img


class FFHQDataset(Dataset):
    def __init__(self, yaml_path, data_dir, split='train', max_load_num=70000):
        """
        Args:
            yaml_path (str): YAML 配置文件路径
            split (str): 数据集 split, 'train', 'val', 或 'test'
            min_subset_size (int): 每次返回子集的最小大小
            max_subset_size (int): 每次返回子集的最大大小（None表示使用所有可用图像）
        """
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)['dataset']  # 加载 YAML 配置

        self.split = split
        self.image_size = self.config.get('image_size', 512)  # 获取 image_size，如果 YAML 中没有则默认为 512
        

        self.max_load_num = max_load_num

        # 存储每个人的所有图片路径
        self.person_images = self._load_person_groups(data_dir, max_load_num)
        
        self.transforms = self._build_transforms(self.config['transforms'][split])  # 构建 transforms
        
        print(f"ffhqDataset初始化完成，{split}集合中共有 {len(self.person_images)} 个人物组")


    def _load_person_groups(self, data_file, max_load_num):
        """从YAML文件加载每个人的图片路径组"""
        person_groups = []
        idx = 0
        for root, dirs, files in os.walk(data_file):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    img_path = os.path.join(root, file)
                    person_groups.append(img_path)
                    idx += 1
                    if idx >= max_load_num:
                        return person_groups
        
        return person_groups


    def _build_transforms(self, transform_config):
        """根据配置构建 transforms"""
        transform_list = []
        for transform_str in transform_config:
            parts = transform_str.split(':')
            transform_name = parts[0].strip()
            params = parts[1].strip() if len(parts) > 1 else None

            if transform_name == 'Resize':
                size = int(params) if params else self.image_size
                transform_list.append(transforms.Resize((size, size)))
            elif transform_name == 'RandomCrop':
                size = int(params) if params else self.image_size
                transform_list.append(transforms.RandomCrop(size))
            elif transform_name == 'CenterCrop':
                size = int(params) if params else self.image_size
                transform_list.append(transforms.CenterCrop(size))
            elif transform_name == 'RandomHorizontalFlip':
                transform_list.append(transforms.RandomHorizontalFlip())
            elif transform_name == 'ToTensor':
                transform_list.append(transforms.ToTensor())
            elif transform_name == 'Normalize':
                mean_str, std_str = params.split('],') if params else ("[0.5]", "[0.5]")  # 默认 Normalize 参数
                mean_str = mean_str.strip() + "]"
                mean = eval(mean_str.strip())  # 使用 eval 解析字符串形式的 list
                std = eval(std_str.strip())
                transform_list.append(transforms.Normalize(mean, std))
            # 可以添加更多transform类型

        return transforms.Compose(transform_list)


    def _extract_label_from_path(self, image_path):
        """从图片路径中提取标签"""
        from pathlib import Path

        path_obj = Path(image_path)
        path_without_extension = str(path_obj.with_suffix(''))

        labels = os.path.basename(path_without_extension).split("_")
        return labels


    def __len__(self):
        return len(self.person_images)


    def __getitem__(self, idx):
        image_path = self.person_images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except ValueError:
            print(image_path,"broken")
            # image = Image.open(self.person_images[0])

        image = resize_and_pad_image(image, output_size=(self.image_size, self.image_size), pad_mode="gray")

        image = self.transforms(image) 

        labels = self._extract_label_from_path(image_path)

        return {
            "images": image,  # 列表形式，每个元素形状为 [C,H,W]
            "original_sizes": torch.tensor([self.image_size, self.image_size]),
            "crop_top_lefts": torch.tensor([0, 0]),
            "age": int(float(labels[0])),
            "gender": labels[1],
            # "race": labels[3],
            "filenames": image_path,
        }

def crop_image_pil(image_path, face_rect, output_path):
    img = Image.open(image_path)
    # 转换坐标格式 (left, upper, right, lower)
    box = (face_rect["left"], 
           face_rect["top"],
           face_rect["left"] + face_rect["width"],
           face_rect["top"] + face_rect["height"])
    cropped = img.crop(box)
    cropped.save(output_path)






