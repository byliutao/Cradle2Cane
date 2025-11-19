import torch
from diffusers import AutoencoderKL
import os
from types import SimpleNamespace
import argparse
from PIL import Image
import time
import re 
from diffusers import StableDiffusionXLImg2ImgPipeline
from lib.utils import train_utils, common_utils, config_utils
from lib.model.arcface import ArcFace

def make_callback(storage_list):
    def callback(pipeline, step, timestep, callback_kwargs):
        emb = callback_kwargs.get("add_text_embeds", None)
        if emb is not None:
            storage_list.append(emb.clone().detach().cpu())
        return {}  
    return callback


def infer_image_with_ID(config, inputs, models, attr_strength, generator=None, weight_dtype=torch.float32):
    # 1. get aged image
    saved_embeddings = []

    models["pipeline"].disable_lora()
    aged_image = models["pipeline"](
        prompt = inputs['prompt'],
        image=inputs['input_image'], 
        num_inference_steps=4, 
        strength=attr_strength,
        guidance_scale=1, 
        generator=generator,
        num_images_per_prompt=1,
        callback_on_step_end=make_callback(saved_embeddings),
        callback_on_step_end_tensor_inputs=["latents", "add_text_embeds"],     
    ).images[0]
    models["pipeline"].enable_lora()


    embedding_list = []
    # 2. get face embedding
    if getattr(config, 'use_arcface_project', False):
        models["face_project_model"].eval()
        if getattr(config, 'use_swr', False):
            if getattr(config, 'use_four_image_swr', False):
                additional_age_pixels = []
                for strength in config.all_strength:
                    if strength == attr_strength:
                        continue
                    models["pipeline"].disable_lora()
                    aged_image_add = models["pipeline"](
                        prompt = inputs['prompt'],
                        image=inputs['input_image'], 
                        num_inference_steps=4, 
                        strength=strength,
                        guidance_scale=1, 
                        generator=generator,
                        num_images_per_prompt=1,
                    ).images[0]
                    models["pipeline"].enable_lora()
                    additional_age_pixels.append([aged_image_add])
            
            else:
                additional_age_pixels = None

            project_face_embedding = train_utils.get_arcface_embedding(config.face_project_use_hidden_state, [inputs["input_image"]], models["arcFace"], models["arcFace"].net, 
                                                        models["face_project_model"], weight_dtype, age_pixels_list=[aged_image], additional_age_pixels=additional_age_pixels, 
                                                        swr_alpha=config.swr_alpha, swr_beta=config.swr_beta)
        else:
            project_face_embedding = train_utils.get_arcface_embedding(config.face_project_use_hidden_state, [inputs["input_image"]], models["arcFace"], models["arcFace"].net, 
                                                        models["face_project_model"], weight_dtype, age_pixels_list=None)
        embedding_list.append(project_face_embedding)
        models["face_project_model"].train()

    # 2.1 get clip embedding
    if getattr(config, 'use_clip_project', False):
    # if args.use_clip_project:
        models["clip_project_model"].eval()
        
        input_pixels_list = [inputs["input_image"]]
        input_ages_list = [inputs["input_attr"]]
        target_ages_list = [inputs["target_attr"]]

        maped_age_clip_embedding = train_utils.map_attr(input_pixels_list, input_ages_list, target_ages_list, models["clip_l_model"], models["clip_l_processor"], models["pipeline"].device, config.clip_map_model)
        clip_face_embedding = models["clip_project_model"](maped_age_clip_embedding)
        clip_face_embedding = clip_face_embedding.unsqueeze(dim=1) 

        embedding_list.append(clip_face_embedding)

        models["clip_project_model"].train()    

    if len(embedding_list) > 0: 
        project_face_embedding = torch.cat(embedding_list, dim=1)
    else:
        project_face_embedding = None

    # 3. generate using aged image embedding
    final_image = models["pipeline"](
        prompt_embeds = project_face_embedding,
        pooled_prompt_embeds = saved_embeddings[0].to(project_face_embedding.device),
        image=aged_image, 
        num_inference_steps=4, 
        strength=config.id_strength,
        guidance_scale=1, 
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]

    return aged_image, final_image


def load_models_for_infer(args, device, weight_dtype, vae=None, arcFace=None, clip_l_model=None, clip_l_processor=None,):
    # Load previous pipeline
    models = {}
    
    if vae is None:
        vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path,).to(device, dtype=weight_dtype)
    
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device)

    # load attention processors
    pipeline.load_lora_weights(args.output_dir,prefix=None)
    

    if clip_l_model is None or clip_l_processor is None:
        # load clip image encoder
        clip_l_model, clip_l_processor = train_utils.load_clip_L(args.clip_L_path, weight_dtype, device)
        clip_l_processor.image_processor._valid_processor_keys = {}
        clip_l_model.eval()
        clip_l_model.requires_grad_(False)



    if arcFace is None:
        arcFace = ArcFace(args.arcface_weight, args.arcface_network, device, weight_dtype, eval=True, require_grad=False)
    

    # load clip project
    if args.use_clip_project:
        models["clip_project_model"] = train_utils.load_clip_project(args, device, weight_dtype)
    else:
        models["clip_project_model"] = None

    if args.use_arcface_project:
        models["face_project_model"] = train_utils.load_face_project(args, device, weight_dtype)
    else:
        models["face_project_model"] = None
        

    models["arcFace"] = arcFace
    models["clip_l_model"] = clip_l_model
    models["clip_l_processor"] = clip_l_processor
    models["pipeline"] = pipeline    

    return models


def single_infer(config, models, weight_dtype, labels, input_image, target_ages, generator=None, save_combine=False):


    if getattr(config, 'prompt_mode', None) is None:
        config.prompt_mode = "normal"

    prompt = common_utils.generate_prompts([target_ages], labels)[0]

    
    print(prompt)
    attr_strength = train_utils.get_age_strength(config, abs(labels["age"]-target_ages), one_threshold=config.one_threshold)
    
    if config.addition_prompt is not None:
        prompt = f"{prompt}, {config.addition_prompt}"
        attr_strength = 0.75
            
    inputs = {"prompt": prompt, "input_image": input_image, "input_attr": labels["age"], "target_attr": target_ages,}
    # print(attr_strength)
    aged_image, final_image = infer_image_with_ID(config, inputs, models, attr_strength, generator, weight_dtype)
    
    if save_combine:
        combined = common_utils.horizontal_concat([input_image, aged_image, final_image])
        return combined
    else:
        return final_image


def stitch_images(images):
    # 计算行数
    num_images = len(images)
    rows = (num_images + 9) // 10
    # 打开第一张图片以获取尺寸
    first_image = images[0]
    width, height = first_image.size

    # 创建一个新的空白图像用于拼接
    stitched_image = Image.new('RGB', (width * 10, height * rows))

    for i in range(num_images):
        # 计算图片在拼接图像中的位置
        row = i // 10
        col = i % 10
        # 将图片粘贴到拼接图像的相应位置
        stitched_image.paste(images[i], (col * width, row * height))

    return stitched_image


def remove_background_with_cravekit(image: Image.Image):
    """
    使用 CraveKit 对图像进行背景剔除
    返回剔除背景后的 PIL.Image 对象
    """
    import torch
    from carvekit.api.high import HiInterface

    # Check doc strings for more information
    interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)
    images_without_background = interface([image])[0].convert('RGB')

    return images_without_background


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age transformation model inference")
    parser.add_argument("--models_dir", type=str, default="models/Cradle2Cane", help="Directory of the models")
    parser.add_argument("--output_dir", type=str, default="outputs/infer", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on")
    parser.add_argument("--weight_dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Data type for weights")
    parser.add_argument("--input_path", type=str, default="asserts/23_male.png", help="Input image path or folder")
    parser.add_argument("--save_combine", type=bool, default=False, help="Whether to save combined image")
    parser.add_argument("--addition_prompt", type=str, default=None, help="prompt")
    parser.add_argument("--one_threshold", action="store_true",)
    parser.add_argument("--use_cravekit", action="store_true", help="Whether to remove background using CraveKit before inference")
    
    args = parser.parse_args()

    def check_filename_format(filename):
        """
        检查文件名格式:
        1. {age}: 可以是整数 (24) 或以 .0 结尾的浮点数 (24.0)，但不接受其他小数 (24.5)
        2. {gender}: 必须是 'male' 或 'female'
        例如: 
          - 25_male.png (Pass)
          - 25.0_male.png (Pass)
          - 25.5_male.png (Fail)
        """
        # 正则解释:
        # ^               : 开头
        # (\d+|\d+\.0)    : 匹配 "纯数字" 或 "数字.0"
        # _               : 下划线
        # (male|female)   : 性别限制
        # \.              : 点
        # (png|jpg|jpeg)$ : 后缀
        pattern = r'^(\d+|\d+\.0)_(male|female)\.(png|jpg|jpeg)$'
        
        return bool(re.match(pattern, filename, re.IGNORECASE))
    # ---------------------------------

    config = config_utils.load_training_config(f"{args.models_dir}/hparams.yml")
    config = SimpleNamespace(**config)
    config.output_dir = args.models_dir
    config.one_threshold = args.one_threshold
    config.addition_prompt = args.addition_prompt
    os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[args.weight_dtype]

    models = load_models_for_infer(config, args.device, weight_dtype)
    target_attrs = [1 + i * 1 for i in range(0, 80)]
    
    base_name_global = os.path.splitext(os.path.basename(args.input_path))[0]
    if config.addition_prompt is not None:
        base_name_global = f"{base_name_global}_{config.addition_prompt}"

    args.output_dir = os.path.join(args.output_dir, base_name_global) 
    os.makedirs(args.output_dir, exist_ok=True)

    def process_image_file(image_path, output_subdir):
        filename = os.path.basename(image_path)
        current_base_name = os.path.splitext(filename)[0]
        
        images = []
        input_image = common_utils.load_and_process_image(image_path)
        labels = common_utils.get_labels_from_path(image_path)
        if args.use_cravekit:
            input_image = remove_background_with_cravekit(input_image)
        
        input_image.save(os.path.join(output_subdir, f"{current_base_name}.png"))

        for i, target_attr in enumerate(target_attrs):
            start_time = time.time()
            result = single_infer(
                config=config,
                models=models,
                weight_dtype=weight_dtype,
                labels=labels,
                input_image=input_image,
                target_ages=target_attr,
                save_combine=args.save_combine,
            )
            images.append(result)
            save_path = os.path.join(output_subdir, f"{current_base_name}_{target_attr}.png")
            result.save(save_path)
            elapsed = time.time() - start_time
            print(f"Inference for {filename} @ {target_attr} took {elapsed:.2f} seconds.")

        stitched = common_utils.stitch_images(images, 10)
        stitched.save(os.path.join(output_subdir, f"{current_base_name}_stitched.png"))

    # --- 主执行逻辑 ---

    if os.path.isdir(args.input_path):
        print(f"Processing directory: {args.input_path}")
        files = os.listdir(args.input_path)
        for filename in files:
            if filename.startswith('.'): 
                continue
                
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                if not check_filename_format(filename):
                    print(f"⚠️  Skipping '{filename}': Invalid format. Name must be '{{number}}_{{male/female}}.ext'")
                    continue
                
                image_path = os.path.join(args.input_path, filename)
                output_subdir = os.path.join(args.output_dir, os.path.splitext(filename)[0])
                os.makedirs(output_subdir, exist_ok=True)
                process_image_file(image_path, output_subdir) # 注意：process_image_file 定义需在上面
    
    elif os.path.isfile(args.input_path):
        filename = os.path.basename(args.input_path)
        
        if not check_filename_format(filename):
            raise ValueError(f"Invalid filename: '{filename}'. \nFormat requirement: {{Age}}_{{Gender}}.png/jpg\nGender must be 'male' or 'female'.")
            
        process_image_file(args.input_path, args.output_dir)
        
    else:
        raise ValueError(f"Invalid input_path: {args.input_path}")

