import torch
from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
import os
from types import SimpleNamespace
import argparse
from PIL import Image
import time

from lib.utils import train_utils, common_utils, config_utils
from lib.model.arcface import ArcFace


def make_callback(storage_list):
    def callback(pipeline, step, timestep, callback_kwargs):
        emb = callback_kwargs.get("add_text_embeds", None)
        if emb is not None:
            storage_list.append(emb.clone().detach().cpu())
        return {}
    return callback


def infer_image_with_ID(config, inputs, models, attr_strength,
                         generator=None, weight_dtype=torch.float32):

    saved_embeddings = []

    # First pass: age transform only
    models["pipeline"].disable_lora()
    aged_image = models["pipeline"](
        prompt=inputs['prompt'],
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

    # ArcFace projection
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
                        prompt=inputs['prompt'],
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

            project_face_embedding = train_utils.get_arcface_embedding(
                config.face_project_use_hidden_state,
                [inputs["input_image"]],
                models["arcFace"],
                models["arcFace"].net,
                models["face_project_model"],
                weight_dtype,
                age_pixels_list=[aged_image],
                additional_age_pixels=additional_age_pixels,
                swr_alpha=config.swr_alpha,
                swr_beta=config.swr_beta
            )

        else:
            project_face_embedding = train_utils.get_arcface_embedding(
                config.face_project_use_hidden_state,
                [inputs["input_image"]],
                models["arcFace"],
                models["arcFace"].net,
                models["face_project_model"],
                weight_dtype,
            )

        embedding_list.append(project_face_embedding)
        models["face_project_model"].train()

    # CLIP projection
    if getattr(config, 'use_clip_project', False):
        models["clip_project_model"].eval()

        input_pixels_list = [inputs["input_image"]]
        input_ages_list = [inputs["input_attr"]]
        target_ages_list = [inputs["target_attr"]]

        mapped_clip = train_utils.map_attr(
            input_pixels_list,
            input_ages_list,
            target_ages_list,
            models["clip_l_model"],
            models["clip_l_processor"],
            models["pipeline"].device,
            config.clip_map_model
        )
        clip_face_embedding = models["clip_project_model"](mapped_clip)
        clip_face_embedding = clip_face_embedding.unsqueeze(1)
        embedding_list.append(clip_face_embedding)

        models["clip_project_model"].train()

    # Merge
    if len(embedding_list) > 0:
        project_face_embedding = torch.cat(embedding_list, dim=1)
    else:
        project_face_embedding = None

    # Second pass: final image synthesis
    final_image = models["pipeline"](
        prompt_embeds=project_face_embedding,
        pooled_prompt_embeds=saved_embeddings[0].to(project_face_embedding.device),
        image=aged_image,
        num_inference_steps=4,
        strength=config.id_strength,
        guidance_scale=1,
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]

    return aged_image, final_image


def load_models_for_infer(args, device, weight_dtype,
                          vae=None, arcFace=None,
                          clip_l_model=None, clip_l_processor=None):

    models = {}

    # Load VAE
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_model_name_or_path
        ).to(device, dtype=weight_dtype)

    # SDXL pipeline
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(device)

    # Load LoRA
    pipeline.load_lora_weights(args.output_dir)

    # Load CLIP-L
    if clip_l_model is None or clip_l_processor is None:
        clip_l_model, clip_l_processor = train_utils.load_clip_L(
            args.clip_L_path, weight_dtype, device
        )
        clip_l_processor.image_processor._valid_processor_keys = {}
        clip_l_model.eval()
        clip_l_model.requires_grad_(False)

    # Load ArcFace
    if arcFace is None:
        arcFace = ArcFace(
            args.arcface_weight,
            args.arcface_network,
            device,
            weight_dtype,
            eval=True,
            require_grad=False
        )

    # Optional projection models
    if args.use_clip_project:
        models["clip_project_model"] = train_utils.load_clip_project(args, device, weight_dtype)
    else:
        models["clip_project_model"] = None

    if args.use_arcface_project:
        models["face_project_model"] = train_utils.load_face_project(args, device, weight_dtype)
    else:
        models["face_project_model"] = None

    models.update({
        "arcFace": arcFace,
        "clip_l_model": clip_l_model,
        "clip_l_processor": clip_l_processor,
        "pipeline": pipeline
    })

    return models


def single_infer(config, models, weight_dtype, labels,
                 input_image, target_ages,
                 generator=None, save_combine=False):

    if getattr(config, 'prompt_mode', None) is None:
        config.prompt_mode = "normal"

    prompt = common_utils.generate_prompts([target_ages], labels)[0]
    attr_strength = train_utils.get_age_strength(config, abs(labels["age"] - target_ages))

    inputs = {
        "prompt": prompt,
        "input_image": input_image,
        "input_attr": labels["age"],
        "target_attr": target_ages,
    }

    aged_image, final_image = infer_image_with_ID(
        config, inputs, models, attr_strength, generator, weight_dtype
    )

    if save_combine:
        return common_utils.horizontal_concat([input_image, aged_image, final_image])
    else:
        return final_image


def process_image_file(image_path, output_subdir, config, models,
                       weight_dtype, target_attrs, args):

    os.makedirs(output_subdir, exist_ok=True)

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    print(f"\n▶ Processing {filename} ...")

    input_image = common_utils.load_and_process_image(image_path)
    labels = common_utils.get_labels_from_path(image_path)

    # Save original
    input_image.save(os.path.join(output_subdir, f"{base_name}.png"))

    results = []

    for target_attr in target_attrs:
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

        results.append(result)
        result.save(os.path.join(output_subdir, f"{base_name}_{target_attr}.png"))

        print(f"  - age {target_attr} done in {time.time() - start_time:.2f}s")

    # 拼接图
    stitched = common_utils.stitch_images(results, 10)
    stitched.save(os.path.join(output_subdir, f"{base_name}_stitched.png"))


# ---------------- Main ----------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models/Cradle2Cane")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weight_dtype", type=str, default="float16")
    parser.add_argument("--input_path", type=str, required=True,
                        help="A single image or a folder of images")
    parser.add_argument("--save_combine", action="store_true")
    args = parser.parse_args()

    config = config_utils.load_training_config(f"{args.models_dir}/hparams.yml")
    config = SimpleNamespace(**config)
    config.output_dir = args.models_dir

    os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }[args.weight_dtype]

    # Load all models
    models = load_models_for_infer(config, args.device, weight_dtype)

    # 0~80 步长 1
    target_attrs = list(range(1, 81))

    # If a folder
    if os.path.isdir(args.input_path):

        for filename in os.listdir(args.input_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):

                input_image_path = os.path.join(args.input_path, filename)
                base_name = os.path.splitext(filename)[0]

                output_subdir = os.path.join(args.output_dir, base_name)
                os.makedirs(output_subdir, exist_ok=True)

                process_image_file(
                    input_image_path, output_subdir,
                    config, models, weight_dtype,
                    target_attrs, args
                )

    # If a single file
    elif os.path.isfile(args.input_path):

        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_subdir = os.path.join(args.output_dir, base_name)
        os.makedirs(output_subdir, exist_ok=True)

        process_image_file(
            args.input_path, output_subdir,
            config, models, weight_dtype,
            target_attrs, args
        )

    else:
        raise ValueError("input_path is invalid")
