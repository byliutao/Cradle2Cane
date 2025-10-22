import os
import random
import yaml
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# ==============================================================================
# 1. 辅助函数
# ==============================================================================

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

# ==============================================================================
# 2. PyTorch 数据集 (Dataset) 类
# ==============================================================================

class FFHQDatasetSort(Dataset):
    def __init__(self, yaml_path, data_dir, split='train', max_load_num=float('inf')):
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)['dataset']

        self.split = split
        self.image_size = self.config.get('image_size', 512)
        self.max_load_num = max_load_num

        self.image_paths = self._load_image_paths(data_dir, self.max_load_num)
        
        # 预处理标签，建立年龄到索引的映射
        print("Preprocessing labels to create age groups...")
        self.age_to_indices = {}
        for idx, path in enumerate(self.image_paths):
            try:
                labels = self._extract_label_from_path(path)
                age = int(float(labels[0]))
                if age not in self.age_to_indices:
                    self.age_to_indices[age] = []
                self.age_to_indices[age].append(idx)
            except (ValueError, IndexError) as e:
                print(f"Skipping file with invalid name format: {path}, error: {e}")
        
        min_samples_per_age = 2  # 至少需要2个样本才能形成一个组
        self.age_to_indices = {
            age: indices for age, indices in self.age_to_indices.items() 
            if len(indices) >= min_samples_per_age
        }

        self.transforms = self._build_transforms(self.config['transforms'][split])
        
        print(f"FFHQDataset initialization complete. Found {len(self.age_to_indices)} age groups for the {split} set.")

    def _load_image_paths(self, data_file, max_load_num):
        image_paths = []
        for root, _, files in os.walk(data_file):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)
                    image_paths.append(img_path)
                    if len(image_paths) >= max_load_num:
                        return image_paths
        return image_paths

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
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening {image_path}: {e}. Returning a placeholder.")
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        image = resize_and_pad_image(image, output_size=(self.image_size, self.image_size), pad_mode="gray")
        image = self.transforms(image) 
        labels = self._extract_label_from_path(image_path)

        return {
            "images": image,
            "original_sizes": torch.tensor([self.image_size, self.image_size]),
            "crop_top_lefts": torch.tensor([0, 0]),
            "age": int(float(labels[0])),
            "gender": labels[1],
            "race": labels[3],
            "filenames": image_path,
        }

# ==============================================================================
# 3. 自定义批次采样器 (BatchSampler) 类
# ==============================================================================

class AgeBatchSampler(Sampler):
    def __init__(self, age_to_indices, batch_size, drop_last=True):
        self.age_to_indices = age_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.ages = list(self.age_to_indices.keys())
        self.num_batches = self._calculate_num_batches()

    def _calculate_num_batches(self):
        count = 0
        for age in self.ages:
            num_samples = len(self.age_to_indices[age])
            if self.drop_last:
                count += num_samples // self.batch_size
            else:
                count += (num_samples + self.batch_size - 1) // self.batch_size
        return count

    def __iter__(self):
        random.shuffle(self.ages)
        all_batches = []
        for age in self.ages:
            indices = self.age_to_indices[age][:]
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch_indices)
        
        random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        return self.num_batches

# ==============================================================================
# 4. 主执行函数
# ==============================================================================


def main():
    """主函数，用于演示和测试"""
    BATCH_SIZE = 4

    # 1. 创建数据集实例
    train_dataset = FFHQDatasetSort("config/ffhq_dataset.yaml", "./models/ffhq_crop_gray_wo_crop_race4", split='train')

    # 2. 创建自定义的批次采样器实例
    train_batch_sampler = AgeBatchSampler(
        age_to_indices=train_dataset.age_to_indices,
        batch_size=BATCH_SIZE,
        drop_last=True
    )

    # 3. 创建 DataLoader
    # 重要提示: 当使用 batch_sampler 时, batch_size, shuffle, sampler, drop_last 必须为默认值
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0 # 在Windows上或简单测试中设为0更稳定
    )

    print("\nStarting DataLoader iteration test...")
    print(f"Total batches to iterate: {len(train_loader)}")
    
    # 4. 迭代并验证
    for i, batch in enumerate(train_loader):
        ages = batch['age']
        
        # 验证批次中的所有年龄是否都相同
        first_age = ages[0].item()
        is_same = all(age.item() == first_age for age in ages)
        
        print(f"Batch {i+1}/{len(train_loader)} | Size: {len(ages)} | Age: {first_age} | All Same: {is_same}")
        
        assert is_same, f"Error: Batch {i+1} contains mixed ages!"
        
    print("\nTest finished successfully! All batches contained images of a single age.")
    print("You can now delete the 'temp_ffhq_data' directory.")


if __name__ == "__main__":
    main()