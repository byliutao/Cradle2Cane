import os
import torch
import argparse
from types import SimpleNamespace
from PIL import Image
from tqdm import tqdm
import numpy as np

# Assuming these are your existing utility modules
from lib.utils import config_utils, common_utils, train_utils
from infer import load_models_for_infer, infer_image_with_ID

def main_generate_aged_dataset_from_folder(
    args
):
    """
    Main function to generate an aged dataset from a folder of images.
    Each image in the input folder is treated as a unique ID.
    Output structure: output_dir / image_filename_as_id / age_XX.jpg
    """
    os.makedirs(args.output_dir, exist_ok=True)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    config_dict = config_utils.load_training_config(os.path.join(args.models_dir, "hparams.yml"))
    config = SimpleNamespace(**config_dict)
    config.output_dir = args.models_dir # Model loading might depend on this path in config
    config.prompt_mode="normal" # Set globally
    config.use_fixed_strength = False # Set globally

    # Determine weight_dtype from string, allow override by training config's mixed_precision
    final_weight_dtype = torch.float32 # Default
    if args.weight_dtype == "float16":
        final_weight_dtype = torch.float16
    elif args.weight_dtype == "bfloat16":
        final_weight_dtype = torch.bfloat16
    
    if hasattr(config, 'mixed_precision'): # Training config takes precedence
        if config.mixed_precision == "fp16":
            final_weight_dtype = torch.float16
        elif config.mixed_precision == "bf16":
            final_weight_dtype = torch.bfloat16
        elif config.mixed_precision == "no" or config.mixed_precision == "fp32": # Explicitly fp32
            final_weight_dtype = torch.float32
    print(f"Using weight dtype: {final_weight_dtype}")

    models = load_models_for_infer(config, args.device, final_weight_dtype)
    if hasattr(models.get("pipeline"), "set_progress_bar_config"):
        models["pipeline"].set_progress_bar_config(disable=True)

    generator = None
    if args.seed is not None:
        # <<< MODIFIED: Ensure each worker has a different seed if a base seed is provided >>>
        worker_seed = args.seed + args.worker_rank if args.seed is not None else None
        if worker_seed is not None:
            generator = torch.Generator(device=args.device).manual_seed(worker_seed)

    target_ages_list = list(range(args.min_target_age, args.min_target_age + args.num_ages_per_id))
    print(f"Targeting {len(target_ages_list)} ages per ID: from {target_ages_list[0]} to {target_ages_list[-1]}")

    if not os.path.isdir(args.input_image_folder):
        print(f"错误: 输入图像文件夹不存在或不是一个目录: {args.input_image_folder}")
        return

    # Get all image files first
    all_image_files = sorted([
        f for f in os.listdir(args.input_image_folder) 
        if os.path.isfile(os.path.join(args.input_image_folder, f)) and \
            any(f.lower().endswith(ext) for ext in image_extensions)
    ])
    
    # <<< MODIFIED: START of code to select a subset of files for this worker >>>
    # Distribute files among workers. Worker with rank `r` gets every `n`-th file.
    # Example: 4 workers (0, 1, 2, 3). Worker 1 gets files 1, 5, 9, ...
    image_files = [
        f for i, f in enumerate(all_image_files) 
        if i % args.num_workers == args.worker_rank
    ]

    print(f"[Worker {args.worker_rank}/{args.num_workers}] Processing {len(image_files)} out of {len(all_image_files)} total images.")
    # <<< MODIFIED: END of code to select a subset of files >>>
    
    if not image_files:
        # It's possible a worker gets no files if there are more workers than files.
        print(f"错误: 在输入文件夹中没有找到该工作进程要处理的图像文件: {args.input_image_folder}")
        return

    ids_processed_count = 0
    # Use tqdm on the worker-specific file list
    for image_filename in tqdm(image_files, desc=f"Worker {args.worker_rank} Processing IDs"):
        if args.num_ids_to_process is not None and ids_processed_count >= args.num_ids_to_process:
            print(f"Reached max number of IDs (images) to process: {args.num_ids_to_process}")
            break

        id_name = os.path.splitext(image_filename)[0] # Use filename (without ext) as ID
        source_image_path = os.path.join(args.input_image_folder, image_filename)
        current_id_output_dir = os.path.join(args.output_dir, id_name)

        # Check if this ID is already complete
        if os.path.isdir(current_id_output_dir):
            try:
                existing_files = [f for f in os.listdir(current_id_output_dir) if f.lower().endswith('.jpg')]
                if len(existing_files) >= args.num_ages_per_id:
                    tqdm.write(f"ID '{id_name}' is already complete ({len(existing_files)} images found). Skipping.")
                    ids_processed_count += 1
                    continue
            except OSError as e:
                tqdm.write(f"Warning: Could not read directory {current_id_output_dir}: {e}. Proceeding to process.")

        try:
            source_image_pil = common_utils.load_and_process_image(source_image_path)
            source_label = common_utils.get_labels_from_path(source_image_path)
        except Exception as e:
            print(f"Error loading or processing source image {source_image_path} (ID: {id_name}): {e}. Skipping.")
            continue

        actual_source_age = source_label.get("age")
        
        os.makedirs(current_id_output_dir, exist_ok=True)

        for target_age_val in tqdm(target_ages_list, desc=f"Ages for ID {id_name}", leave=False):
            prompt = common_utils.generate_prompts([target_age_val], source_label)[0]
            
            try:
                current_source_age_int = int(actual_source_age)
                age_diff = abs(current_source_age_int - target_age_val)
                attr_strength = train_utils.get_age_strength(config, age_diff)
            except ValueError:
                print(f"Warning: Could not parse source_age '{actual_source_age}' as int for ID {id_name}. Using default age for strength calc.")
                age_diff = abs(int(args.default_source_age) - target_age_val)
                attr_strength = train_utils.get_age_strength(config, age_diff)

            inputs = {
                "prompt": prompt,
                "input_image": source_image_pil,
                "input_attr": actual_source_age, 
                "target_attr": target_age_val,
            }

            try:
                _, final_output_image_pil = infer_image_with_ID(config, inputs, models, attr_strength, generator, final_weight_dtype)
                
                if args.output_resolution and args.output_resolution > 0:
                    final_output_image_pil = final_output_image_pil.resize(
                        (args.output_resolution, args.output_resolution), 
                        Image.LANCZOS
                    )

                output_filename = f"age_{target_age_val:02d}.jpg" 
                output_save_path = os.path.join(current_id_output_dir, output_filename)
                final_output_image_pil.save(output_save_path)
            except Exception as e:
                print(f"Error during inference or saving for ID {id_name}, target age {target_age_val}: {e}")
                continue 
        
        ids_processed_count += 1
    
    print(f"Worker {args.worker_rank} finished generating its portion of the aged dataset. Processed {ids_processed_count} image files (IDs).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从图像文件夹生成伪年龄数据集 (每个图像是一个ID)")

    parser.add_argument("--input_image_folder", type=str, required=True, help="包含源图像的文件夹 (每个图像文件代表一个ID, 无子目录)")
    parser.add_argument("--models_dir", type=str, default="models/cradle_2", help="训练模型目录，需包含 hparams.yml")
    parser.add_argument("--output_dir", type=str, default="temp/generated_fake_aged_dataset", help="伪造数据集的输出路径")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="模型运行设备")
    parser.add_argument("--weight_dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"], help="模型权重精度类型 (会被训练配置中的 mixed_precision 覆盖)")
    parser.add_argument("--seed", type=int, default=None, help="PyTorch Generator seed for reproducibility")
    parser.add_argument("--num_ids_to_process", type=int, default=None, help="要处理的输入图像文件数量上限 (默认处理所有)")
    parser.add_argument("--min_target_age", type=int, default=15, help="生成的最小目标年龄")
    parser.add_argument("--num_ages_per_id", type=int, default=50, help="每个身份ID (源图像) 生成的目标年龄图像数量 (例如15-64岁)")
    parser.add_argument("--output_resolution", type=int, default=112, help="保存生成图像的分辨率 (例如 112 for 112x112). 设置为 <=0 则保存模型原始输出分辨率.")

    # <<< MODIFIED: Add worker arguments >>>
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of parallel workers.")
    parser.add_argument("--worker_rank", type=int, default=0, help="Rank of this worker (from 0 to num_workers-1).")

    args = parser.parse_args()
    
    main_generate_aged_dataset_from_folder(args)