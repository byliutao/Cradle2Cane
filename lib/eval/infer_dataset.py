import os
import torch
import argparse
from types import SimpleNamespace
from PIL import Image
from tqdm import tqdm

from lib.utils import config_utils, common_utils, train_utils
from infer import load_models_for_infer, infer_image_with_ID

def process_single_image(config, image_path, models, output_base_dir, weight_dtype, target_ages, generator=None, save_combine=False):
    # 加载输入图像及标签
    input_image = common_utils.load_and_process_image(image_path)
    label = common_utils.get_labels_from_path(image_path)

    for target_attr in target_ages:
        prompt = common_utils.generate_prompts([target_attr], label,)[0]
        attr_strength = train_utils.get_age_strength(config, abs(label["age"] - target_attr), one_threshold=config.one_threshold)

        inputs = {
            "prompt": prompt,
            "input_image": input_image,
            "input_attr": label["age"],
            "target_attr": target_attr,
        }

        aged_image, final_image = infer_image_with_ID(config, inputs, models, attr_strength, generator, weight_dtype)

        output_dir = os.path.join(output_base_dir, f"{target_attr}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        if os.path.exists(output_path):
            print(f"跳过已存在的文件: {output_path}")
            break

        if save_combine:
            combined = common_utils.horizontal_concat([input_image, aged_image, final_image])
            combined.save(output_path)
        else:
            final_image.save(output_path)

def main_folder(input_folder, models_dir, output_dir, device, weight_dtype, target_ages,
                  max_num=20, generator=None, save_combine=False, one_threshold=False):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    # 读取训练配置
    config = config_utils.load_training_config(os.path.join(models_dir, "hparams.yml"))
    config = SimpleNamespace(**config)
    config.output_dir = models_dir
    config.one_threshold = one_threshold

    # 自动推断数据类型
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 加载所有模型组件
    models = load_models_for_infer(config, device, weight_dtype)
    models["pipeline"].set_progress_bar_config(disable=True)

    # 遍历输入文件夹中的图像文件
    for idx, filename in enumerate(tqdm(sorted(os.listdir(input_folder)))):
        if idx >= max_num:
            break
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(input_folder, filename)
            process_single_image(
                config=config,
                image_path=image_path,
                models=models,
                output_base_dir=output_dir,
                weight_dtype=weight_dtype,
                target_ages=target_ages,
                generator=generator,
                save_combine=save_combine,
            )

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="批量年龄编辑推理脚本")

    parser.add_argument("--models_dir", type=str, default="models/Cradle2Cane", help="训练模型目录，需包含 hparams.yml")
    parser.add_argument("--output_dir", type=str, default="temp/agedb-400-label", help="输出图像路径")
    parser.add_argument("--input_folder", type=str, default="models/agedb-400-label", help="原始图像文件夹")
    parser.add_argument("--device", type=str, default="cuda:0", help="模型运行设备")
    parser.add_argument("--weight_dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"], help="模型权重精度类型")
    parser.add_argument("--one_threshold", action="store_true",)
    parser.add_argument("--save_combine", action="store_true", help="是否保存原图+编辑图+最终图拼接结果")
    parser.add_argument("--max_num", type=int, default=1000, help="每类最多处理图像数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于确保结果可复现") # <-- 1. 新增 seed 参数

    args = parser.parse_args()

    # dtype 转换
    torch_dtype = getattr(torch, args.weight_dtype)

    # 2. 创建带指定种子的 Generator 对象
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    target_ages = [20, 25, 30, 35, 40, 45, 50, 65, 70]
    print(target_ages)

    # 处理整文件夹
    main_folder(
        input_folder=args.input_folder,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        device=args.device,
        weight_dtype=torch_dtype,
        target_ages=target_ages,
        max_num=args.max_num,
        save_combine=args.save_combine,
        generator=generator, 
        one_threshold=args.one_threshold,
    )