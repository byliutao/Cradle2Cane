# modify from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py

"""Fine-tuning script for Stable Diffusion XL for text2image with support for LoRA."""

import logging
import os
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from huggingface_hub import create_repo, upload_folder
from diffusers.optimization import get_scheduler
import wandb
import math

import datasets
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from transformers import AutoTokenizer, PretrainedConfig, CLIPProcessor

import diffusers
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import open_clip
from transformers import AutoProcessor, CLIPModel

from lib.dataset.ffhq_dataset import FFHQDataset
from lib.dataset.ffhq_dataset_sort import FFHQDatasetSort, AgeBatchSampler
from lib.model.sdxl_pipe import retrieve_timesteps
from lib.utils.config_utils import parse_args
from lib.model.clip_model import MY_CLIP_MODEL
from lib.model.embed_fuse import EmbeddingTransformer
from lib.model.face_project import FaceToTextEmbeddingMapping, CLIPToTextEmbeddingMapping
from infer import infer_image_with_ID, load_models_for_infer
from lib.arcface import get_model
from lib.utils import common_utils
from lib.model.age_map import map_attr, map_age1
from lib.model.gan_loss import Discriminator
from lib.model import lipis
from lib.utils import common_utils, train_utils, config_utils
from lib.model.age_predictor import AgePredictor
from lib.model.arcface import ArcFace
from lib.model.gan_loss import Discriminator, generator_hinge_loss, discriminator_hinge_loss


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    train_text_encoder: bool = False,
    repo_folder: str = None,
    vae_path: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def load_sub_modules(
    pretrained_model_name_or_path,
    revision=None,
    variant=None,
    pretrained_vae_model_name_or_path=None,
):
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision, variant=variant
    )
    vae_path = (
        pretrained_model_name_or_path
        if pretrained_vae_model_name_or_path is None
        else pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if pretrained_vae_model_name_or_path is None else None,
        revision=revision,
        # variant=variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant
    )

    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, unet, noise_scheduler


def create_optimizer(args, models, train_models_name):
    """
    Create an optimizer for training the models.
    """
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs

    optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = []
    for model_name in train_models_name:
        params_to_optimize.extend(list(filter(lambda p: p.requires_grad, models[model_name].parameters())))
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    return optimizer, params_to_optimize


def setup_accelerator_and_logging(args, logger):
    """
    设置 accelerator 和日志记录
    """
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
            
    return accelerator


def load_and_setup_models(args, accelerator, logger):
    """
    加载和设置模型
    """
    tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, unet, noise_scheduler = load_sub_modules(
        args.pretrained_model_name_or_path, args.revision, args.variant, args.pretrained_vae_model_name_or_path)
    
    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)

    if args.pretrained_vae_model_name_or_path is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

        
    return tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, unet, noise_scheduler, weight_dtype


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_model_by_name(accelerator, model, output_dir, save_name):
    model_to_save = unwrap_model(accelerator, model)
    model_to_save = model_to_save.state_dict()
    torch.save(model_to_save,os.path.join(output_dir, save_name))


def load_model_by_name(input_dir, load_name, model):
    ckpt_path = os.path.join(input_dir, load_name)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict) 

    return model


def setup_save_and_load_hooks(args, accelerator, input_models, logger):
    """
    设置保存和加载模型的钩子
    """
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for idx, model in enumerate(models):
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, input_models["unet"]))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                    
                elif isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, input_models['text_encoder_one']))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )

                elif isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, input_models['text_encoder_two']))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )

                elif isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, input_models['face_project_model']))):
                    save_model_by_name(accelerator, model, output_dir, args.face_project_save_name)

                elif isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, input_models['clip_project_model']))):
                    save_model_by_name(accelerator, model, output_dir, args.clip_project_save_name)

                else:
                    # raise ValueError(f"unexpected save model: {model.__class__}")
                    pass

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLImg2ImgPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )


    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None
        load_models = []
        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(accelerator, input_models["unet"]))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(accelerator, input_models['text_encoder_one']))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(accelerator, input_models['text_encoder_two']))):
                text_encoder_two_ = model
            elif isinstance(model, type(unwrap_model(accelerator, input_models['embeds_fuse_model']))):
                model = load_model_by_name(input_dir, args.embedding_fuse_save_name, model)
                load_models.append(model)   
            elif isinstance(model, type(unwrap_model(accelerator, input_models['face_project_model']))):
                model = load_model_by_name(input_dir, args.face_project_save_name, model)
                load_models.append(model)   
            elif isinstance(model, type(unwrap_model(accelerator, input_models['clip_project_model']))):
                model = load_model_by_name(input_dir, args.clip_project_save_name, model)
                load_models.append(model)   
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )
        
        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            load_models.append(unet_)
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def prepare_training_environment(args, logger):
    """设置训练环境，包括加速器、随机种子和输出目录"""
    accelerator = setup_accelerator_and_logging(args, logger)

    if accelerator.num_processes > 1:
        torch.distributed.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])                          

    if args.seed is not None:
        accelerator.wait_for_everyone()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        import random
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)  # unet have upsample, cann't set 

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
            return accelerator, repo_id
    
    return accelerator, None


def prepare_dataset_and_dataloader(args, accelerator):
    """准备训练数据集和数据加载器"""
    if os.path.basename(args.dataset_name) == "ffhq_dataset.yaml":
        if args.sort is False:
            train_dataset = FFHQDataset(args.dataset_name, args.train_data_list, split='train', max_load_num=args.max_load_num)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=args.train_batch_size, 
                num_workers=args.dataloader_num_workers, 
                pin_memory=True,
            )
        else:
            # 1. 创建数据集实例
            train_dataset = FFHQDatasetSort(args.dataset_name, args.train_data_list, split='train', max_load_num=args.max_load_num)

            # 2. 创建自定义的批次采样器实例
            train_batch_sampler = AgeBatchSampler(
                age_to_indices=train_dataset.age_to_indices,
                batch_size=args.train_batch_size,
                drop_last=True
            )

            # 3. 创建 DataLoader
            # 重要提示: 当使用 batch_sampler 时, batch_size, shuffle, sampler, drop_last 必须为默认值
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=args.dataloader_num_workers, 
                pin_memory=True,
            )
    else:
        exit("require dataset_name")

    
    return train_dataset, train_dataloader


def calculate_training_steps(args, train_dataloader):
    """计算训练步数和相关参数"""
    overrode_max_train_steps = False
    # print(len(train_dataloader))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    return num_update_steps_per_epoch, overrode_max_train_steps


def prepare_optimizer_and_scheduler(args, models, train_models_name, accelerator):
    """准备优化器和学习率调度器"""
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    optimizer, params_to_optimize = create_optimizer(args, models, train_models_name)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    return optimizer, params_to_optimize, lr_scheduler


def prepare_for_training(args, accelerator, models, train_models_name, optimizer, 
                         train_dataloader, lr_scheduler, num_update_steps_per_epoch, overrode_max_train_steps):
    """准备训练，包括加速器准备和训练步数计算"""
    # Prepare everything with our `accelerator`
    for model_name in train_models_name:
        models[model_name] = accelerator.prepare(models[model_name])    

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler
        )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # 重新计算训练轮数
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # 初始化跟踪器
    filtered_config = {k: v for k, v in vars(args).items() if not isinstance(v, list)}
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=filtered_config)
        
        import yaml
        with open(f'{args.output_dir}/hparams.yml', 'w') as f:
            yaml.dump(filtered_config, f, sort_keys=False, allow_unicode=True)
    
    # 返回所有被accelerator.prepare()处理过的对象
    return models, optimizer, train_dataloader, lr_scheduler, num_update_steps_per_epoch


def resume_from_checkpoint(args, accelerator, num_update_steps_per_epoch):
    """从检查点恢复训练"""
    global_step = 0
    first_epoch = 0
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
        first_epoch = 0
    
    return global_step, first_epoch, initial_global_step


def save_model_weights(args, accelerator, models, train_models_name):
    """保存LoRA权重"""
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    
    unet = unwrap_model(accelerator, models['unet'])
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    

    if args.train_text_encoder:
        text_encoder_one = unwrap_model(accelerator, models['text_encoder_one'])
        text_encoder_two = unwrap_model(accelerator, models['text_encoder_two'])

        text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_one))
        text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_two))
    else:
        text_encoder_lora_layers = None
        text_encoder_2_lora_layers = None

    StableDiffusionXLImg2ImgPipeline.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_layers,
        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
    )

    if 'embeds_fuse_model' in train_models_name:
        save_sub_model_weights(accelerator, args.output_dir, args.embedding_fuse_save_name, models['embeds_fuse_model'])

    if 'face_project_model' in train_models_name:
        save_sub_model_weights(accelerator, args.output_dir, args.face_project_save_name, models['face_project_model'])

    if 'clip_project_model' in train_models_name:
        save_sub_model_weights(accelerator, args.output_dir, args.clip_project_save_name, models['clip_project_model'])  

    del models['unet']
    del models['text_encoder_one']
    del models['text_encoder_two']
    del unet_lora_state_dict
    del text_encoder_lora_layers
    del text_encoder_2_lora_layers
    
    torch.cuda.empty_cache()


def save_sub_model_weights(accelerator, output_dir, save_name, model):
    model_to_save = unwrap_model(accelerator, model)
    model_to_save = model_to_save.state_dict()
    torch.save(
        model_to_save,
        os.path.join(output_dir, save_name)
    )
    del model
    del model_to_save      


def perform_validation(args, accelerator, logger, global_step, models, weight_dtype):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
    )
    # create pipeline
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=models['vae'],
        text_encoder=unwrap_model(accelerator, models['text_encoder_one']),
        text_encoder_2=unwrap_model(accelerator, models['text_encoder_two']),
        unet=unwrap_model(accelerator, models['unet']),
        revision=args.revision,
        variant=args.variant,
        dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    
    # run inference
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        images = run_validation(args, accelerator, models, pipeline, weight_dtype)

    report_to_tracker(args, accelerator.trackers, images[0], images, global_step, "validation")

    del pipeline
    torch.cuda.empty_cache()


def run_validation(args, accelerator, models, pipeline, weight_dtype):
    images = []
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    validation_target_attrs = args.validation_target_age
            
    for input_image_path, target_attr in zip(args.validation_image, validation_target_attrs):
        labels = common_utils.get_labels_from_path(input_image_path)
        input_image = common_utils.load_and_process_image(input_image_path)

        prompt = common_utils.generate_prompts([target_attr], labels)[0]
          

        if args.use_arcface_project:
            face_project_model = unwrap_model(accelerator, models['face_project_model'])
        else:
            face_project_model = None

        if args.use_clip_project:
            clip_project_model = unwrap_model(accelerator, models['clip_project_model'])
        else:
            clip_project_model = None
        
        attr_strength = train_utils.get_age_strength(args, abs(labels["age"]-target_attr), one_threshold=args.one_threshold)

                                            
        inputs = {"prompt": prompt, "input_image": input_image, "input_attr": labels["age"], "target_attr": target_attr,}

        input_models = {"pipeline": pipeline, "arcFace": models['arcFace'], "face_project_model": face_project_model, 
                        "clip_l_model": models["clip_l_model"], "clip_l_processor": models["clip_l_processor"], "clip_project_model": clip_project_model}
        
        aged_image, final_image = infer_image_with_ID(args, inputs, input_models, attr_strength, generator=generator, weight_dtype=weight_dtype, )
        
        images.extend([input_image, aged_image, final_image])

    return images


def report_to_tracker(args, trackers, input_image, images, global_step, name):
    for tracker in trackers:
        if tracker.name == "tensorboard":
            # Include the input image with the generated images
            # all_images = [input_image] + images
            all_images = images
            np_images = np.stack([np.asarray(img) for img in all_images])

            tracker.writer.add_images(name, np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
            # Include the input image with the generated images
            tracker.log(
                {
                    name: [
                        wandb.Image(input_image, caption=f"Input: {args.validation_image}")
                    ] + [
                        wandb.Image(image, caption=f"{i+1}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )


def load_face_project(args, device, weight_dtype):
    if args.face_project_use_hidden_state is True:
        face_project_input_dim = args.face_project_hidden_dim
    else:
        face_project_input_dim = args.face_project_feature_dim
    
    face_project_model = FaceToTextEmbeddingMapping(input_dim=face_project_input_dim, output_dim=args.face_project_output_dim)
    ckpt_path = os.path.join(args.output_dir, args.face_project_save_name)
    state_dict = torch.load(ckpt_path, map_location=device)
    face_project_model.load_state_dict(state_dict)
    face_project_model.to(device, dtype=weight_dtype)
    
    return face_project_model


def load_clip_project(args, device, weight_dtype):
    clip_project_model = CLIPToTextEmbeddingMapping(input_dim=args.clip_project_input_dim, output_dim=args.clip_project_output_dim)
    ckpt_path = os.path.join(args.output_dir, args.clip_project_save_name)
    state_dict = torch.load(ckpt_path,map_location=device)
    clip_project_model.load_state_dict(state_dict)
    clip_project_model.to(device, dtype=weight_dtype)

    return clip_project_model


def perform_final_inference(args, accelerator, models, weight_dtype, repo_id=None):
    """执行最终推理，生成样本并上传到Hub（如果需要）"""
    if not accelerator.is_main_process:
        return
    
    input_models = load_models_for_infer(args, accelerator.device, weight_dtype, models['vae'], 
                                         models['arcFace'], models['clip_l_model'], models['clip_l_processor'])

    # # Make sure vae.dtype is consistent with the unet.dtype

    # run inference
    images = []
    if args.validation_image and args.num_validation_images > 0:
        
        images = run_validation(args, accelerator, input_models, input_models["pipeline"], weight_dtype)

        report_to_tracker(args, accelerator.trackers, images[0], images, 0, "test")

    if args.push_to_hub and repo_id:
        save_model_card(
            repo_id,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            dataset_name=args.dataset_name,
            train_text_encoder=args.train_text_encoder,
            repo_folder=args.output_dir,
            vae_path=args.pretrained_vae_model_name_or_path,
        )
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )



def get_timesteps(scheduler, num_inference_steps, strength, device, denoising_start=None):
    """
    获取扩散过程中使用的时间步
    
    Args:
        scheduler: 扩散过程的调度器
        num_inference_steps: 推理步数
        strength: 噪声强度 (0-1)
        device: 计算设备
        denoising_start: 可选的去噪起始点
    
    Returns:
        tuple: (timesteps, actual_steps)
    """
    # get the original timestep using init_timestep
    if denoising_start is None:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start * scheduler.order)

        return timesteps, num_inference_steps - t_start

    else:
        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        discrete_timestep_cutoff = int(
            round(
                scheduler.config.num_train_timesteps
                - (denoising_start * scheduler.config.num_train_timesteps)
            )
        )

        num_inference_steps = (scheduler.timesteps < discrete_timestep_cutoff).sum().item()
        if scheduler.order == 2 and num_inference_steps % 2 == 0:
            # if the scheduler is a 2nd order scheduler we might have to do +1
            # because `num_inference_steps` might be even given that every timestep
            # (except the highest one) is duplicated. If `num_inference_steps` is even it would
            # mean that we cut the timesteps in the middle of the denoising step
            # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
            # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
            num_inference_steps = num_inference_steps + 1

        # because t_n+1 >= t_n, we slice the timesteps starting from the end
        t_start = len(scheduler.timesteps) - num_inference_steps
        timesteps = scheduler.timesteps[t_start:]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start)
        return timesteps, num_inference_steps


# wo_batch = punish_wight(wo_batch.T.to(float), wo_batch.size(0), alpha=alpha, beta=beta, calc_similarity=False).T.to(prompt_embeds.dtype)
def punish_wight(tensor, latent_size, alpha=1.0, beta=1.2):
    u, s, vh = torch.linalg.svd(tensor)
    u = u[:,:latent_size]
    s *= torch.exp(alpha*s) * beta
    tensor = u @ torch.diag(s) @ vh
    return tensor


def unet_forward(timesteps, prompt_embeds, image_processor, weight_dtype, vae, noise_scheduler, 
                 unet_added_conditions, unet, latents):   # input_pixels must be [-1,1]
    
    for i, t in enumerate(timesteps):

        latent_model_input = noise_scheduler.scale_model_input(latents, t)

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]

        if i == 0:
            noise_pred_init = noise_pred

        latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents.to(weight_dtype), return_dict=False)[0] # image has gradient if latents has gradient

    # convert image from (-1,1) to (0, 1)
    images = image_processor._denormalize_conditionally(images) 

    assert (torch.max(images) <= 1.0 and torch.min(images) >= 0.0)
    return noise_pred_init, images


def get_timesteps_strength(noise_scheduler, num_inference_steps, strength, device):
    timesteps, num_inference_steps = retrieve_timesteps(
        noise_scheduler, num_inference_steps, device
    )
    # get update timesteps by strength
    timesteps, num_inference_steps = get_timesteps(
        noise_scheduler, num_inference_steps, strength, device
    )

    return timesteps    


def get_init_latents(noises, noise_scheduler, model_inputs, num_inference_steps, strength):
    # Sample noise that we'll add to the latents
    timesteps = get_timesteps_strength(noise_scheduler, num_inference_steps, strength, model_inputs.device)

    latent_timestep = timesteps[:1]
    noise_scheduler._begin_index = None # important, because train_utils.get_timesteps set _begin_index TODO: use or not use
    latents = noise_scheduler.add_noise(model_inputs, noises, latent_timestep)

    return timesteps, latents


def get_init_latents_age(noises, noise_scheduler, model_inputs, num_inference_steps, age_diff):
    # get update timesteps by strength
    timesteps = [diff*25 + 100 for diff in age_diff]

    sigmas = np.array(((1 - noise_scheduler.alphas_cumprod) / noise_scheduler.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas).to(device=model_inputs.device, dtype=model_inputs.dtype)
    
    timesteps = torch.tensor(timesteps).to(device=model_inputs.device, dtype=model_inputs.dtype)

    noise_scheduler.sigmas = sigmas

    noise_scheduler.timesteps = timesteps

    latents = model_inputs + noises * sigmas[0]


    return timesteps, latents


def split_prompt(prompt, tokenizer):
    """
    Split a prompt into two parts if its tokenized length exceeds 77
    
    Args:
        prompt: The text prompt to split
        
    Returns:
        A tuple of (first_prompt, second_prompt) where:
        - first_prompt is the first 77 tokens (or the whole prompt if ≤77)
        - second_prompt is the remainder (or None if prompt length ≤77)
    """
    # Tokenize the prompt
    tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    
    # If prompt length doesn't exceed 77, return original prompt and None
    max_length = 77 - 1
    if len(tokens) <= max_length:
        return prompt, None
    
    # Split at the 77th token
    first_tokens = tokens[:max_length]
    second_tokens = tokens[max_length:]
    
    # Decode the tokens back to text
    first_prompt = tokenizer.decode(first_tokens, skip_special_tokens=True)
    second_prompt = tokenizer.decode(second_tokens, skip_special_tokens=True)
    
    return first_prompt, second_prompt


def load_clip_big_G(pretrained_path, weight_dtype, device):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained=pretrained_path)
    model = model.to(device, dtype=weight_dtype)

    return model, preprocess


def load_clip_L(pretrained_path, weight_dtype, device):
    model = MY_CLIP_MODEL.from_pretrained(pretrained_path)
    processor = CLIPProcessor.from_pretrained(pretrained_path, use_fast=True)
    model = model.to(device, dtype=weight_dtype)

    return model, processor


def encode_image(clip_L: MY_CLIP_MODEL, clip_big_G, pixel_values):
    image_embeds_clip_L, _ = clip_L.get_image_embeds(pixel_values=pixel_values, output_hidden_states=True)
    
    clip_big_G.visual.output_tokens = True
    pooled_embeds, image_embeds_clip_big_G = clip_big_G.encode_image(pixel_values)
    image_embeds_clip_big_G = image_embeds_clip_big_G @ clip_big_G.visual.proj


    return image_embeds_clip_L, image_embeds_clip_big_G, pooled_embeds


def load_all_models(args, accelerator, logger):
    models = {}

    # Load sdxl_turbo
    (tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae,
    unet, noise_scheduler, weight_dtype) = load_and_setup_models(args, accelerator, logger)
    
    
    # load clip image encoder
    clip_l_model, clip_l_processor = load_clip_L(args.clip_L_path, weight_dtype, accelerator.device)
    clip_l_model.eval()
    clip_l_model.requires_grad_(False)
    

    use_fp16 = (True if weight_dtype is torch.float16 else False)
    arcface_net = get_model(args.arcface_network, fp16=use_fp16).to(accelerator.device)
    arcface_net.load_state_dict(torch.load(args.arcface_weight,))
    arcface_net.eval()
    arcface_net.requires_grad_(False)

    if args.face_project_use_hidden_state is True:
        face_project_input_dim = args.face_project_hidden_dim
    else:
        face_project_input_dim = args.face_project_feature_dim

    if args.use_arcface_project:
        face_project_model = FaceToTextEmbeddingMapping(input_dim=face_project_input_dim, output_dim=args.face_project_output_dim)
    else:
        face_project_model = None

    if args.use_clip_project:
        clip_project_model = CLIPToTextEmbeddingMapping(input_dim=args.clip_project_input_dim, output_dim=args.clip_project_output_dim)
    else:
        clip_project_model = None

    loss_fn_alex = lipis.FIX_LPIPS(net=args.lpips_net, pretrained=True, model_path=args.lpips_model_path, lpips=True,
                                    use_dropout=False, eval_mode=True).to(accelerator.device, dtype=torch.float32)
    agePredictor = AgePredictor(args.age_model_path, accelerator.device, torch.float32, eval=True, require_grad=False)
    arcFace = ArcFace(args.arcface_weight, args.arcface_network, accelerator.device, torch.float32, eval=True, require_grad=False)
    
    if args.use_gan_loss:
        D = Discriminator(in_channels=3).to(accelerator.device, dtype=torch.float32)
    else:
        D = None

    models['tokenizer_one'] = tokenizer_one
    models['tokenizer_two'] = tokenizer_two
    models['text_encoder_one'] = text_encoder_one
    models['text_encoder_two'] = text_encoder_two
    models['vae'] = vae
    models['unet'] = unet
    models['noise_scheduler'] = noise_scheduler
    models['clip_l_model'] = clip_l_model
    models['clip_l_processor'] = clip_l_processor
    models['arcface_model'] = arcface_net
    models['face_project_model'] = face_project_model
    models['clip_project_model'] = clip_project_model
    models["arcFace"] = arcFace
    models["agePredictor"] = agePredictor
    models["loss_fn_alex"] = loss_fn_alex
    models["D"] = D     


    train_models_name_G = ["unet",]
    if args.train_text_encoder:
        train_models_name_G.extend(["text_encoder_one","text_encoder_two"])

    if args.use_arcface_project:
        train_models_name_G.extend(["face_project_model",])

    if args.use_clip_project:
        train_models_name_G.extend(["clip_project_model",])


    return models, train_models_name_G, weight_dtype


def compute_time_ids(resolution, original_size, crops_coords_top_left, device, weight_dtype):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = (resolution, resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.to(device, dtype=weight_dtype)
    return add_time_ids


def get_age_strength(args, age_diff, one_threshold=False):
    if one_threshold:
        if abs(age_diff) <= args.t1:
            aged_strength = 0.25
        else:
            aged_strength = 0.5

        return aged_strength
    
    if args.use_adaptive_noise_inject:
        if abs(age_diff) <= args.t1:
            aged_strength = 0.25
        elif abs(age_diff) <= args.t2:
            aged_strength = 0.5
        else:
            aged_strength = 0.75

        return aged_strength
    else:
        return args.aged_strength


def get_inputs_from_batch_ffhq(args, batch, accelerator, weight_dtype):
    batch_size = len(batch["images"])
    input_pixels = batch["images"]  # value: 0~1
    input_ages = batch["age"] # Per your request, all ages in this batch are the same.
    # print(input_ages)
    # 1. 为整个批次生成一个初始的、期望的年龄差 (initial_age_diff)
    if torch.rand(1) > args.same_age_rate:
        if args.age_sample_mode == "normal":
            # 正态分布采样
            normal_tensor = torch.randn(1) * args.age_scale
            clamped_tensor = torch.clamp(normal_tensor, min=-3 * args.age_scale, max=3 * args.age_scale)
            initial_age_diff = clamped_tensor.round().int().to(accelerator.device)

        elif args.age_sample_mode == "uniform":
            # 均匀分布采样
            max_diff = args.age_scale * 3
            initial_age_diff = torch.randint(
                low=-max_diff, high=max_diff + 1, size=(1,)
            ).to(accelerator.device)
        else:
            raise ValueError(f"Unsupported age_sample_mode: {args.age_sample_mode}")
    else:
        # 保持原年龄
        initial_age_diff = torch.zeros(1, dtype=torch.int32).to(accelerator.device)

    # 2. 计算目标年龄，并立即使用 torch.clamp 进行向量化裁剪，确保年龄在有效范围内
    target_ages = torch.clamp(input_ages + initial_age_diff, min=args.min_age, max=args.max_age)

    # 3. 计算裁剪后实际生效的年龄差 (actual_age_diff)
    actual_age_diff = target_ages[0] - input_ages[0]

    aged_strength = get_age_strength(args, actual_age_diff.item(), args.one_threshold)

    # print(target_ages, input_ages, actual_age_diff, aged_strength)
    
    target_face_descriptions = common_utils._generate_prompts(target_ages, batch["gender"])

    original_sizes = batch["original_sizes"]
    crop_top_lefts = batch["crop_top_lefts"]
    target_sizes = torch.tensor([[args.resolution, args.resolution]] * batch_size, device=accelerator.device, dtype=torch.long)
    add_time_ids = torch.cat((original_sizes, crop_top_lefts, target_sizes), dim=1).to(dtype=weight_dtype)

    # 将 Tensor 转换为 list
    input_ages_list = [age for age in input_ages]
    target_ages_list = [age for age in target_ages]

    return {
        "input_pixels": input_pixels, 
        "target_face_descriptions": target_face_descriptions, 
        "input_attrs": input_ages_list,
        "target_attrs": target_ages_list, 
        "aged_strength": aged_strength, 
        "age_diff": actual_age_diff, # 返回实际的 age_diff
        "add_time_ids": add_time_ids
    }


def get_inputs_from_batch_ffhq_avg(args, batch, accelerator, weight_dtype):
    batch_size = len(batch["images"])
    input_pixels = batch["images"]  # value: 0~1
    input_ages = batch["age"] # 这里的 age 是一个包含不同年龄的 Tensor

    # 1. 为整个批次生成一个统一的、期望的年龄差 (initial_age_diff)
    if torch.rand(1) > args.same_age_rate:
        if args.age_sample_mode == "normal":
            normal_tensor = torch.randn(1) * args.age_scale
            clamped_tensor = torch.clamp(normal_tensor, min=-3 * args.age_scale, max=3 * args.age_scale)
            initial_age_diff = clamped_tensor.round().int().to(accelerator.device)
        elif args.age_sample_mode == "uniform":
            max_diff = args.age_scale * 3
            initial_age_diff = torch.randint(
                low=-max_diff, high=max_diff + 1, size=(1,)
            ).to(accelerator.device)
        else:
            raise ValueError(f"Unsupported age_sample_mode: {args.age_sample_mode}")
    else:
        initial_age_diff = torch.zeros(1, dtype=torch.int32).to(accelerator.device)

    # 2. 计算目标年龄，并进行向量化裁剪
    target_ages = torch.clamp(input_ages + initial_age_diff, min=args.min_age, max=args.max_age)


    # 3. 计算批次中【每个样本】的实际年龄差，得到一个 Tensor
    actual_age_diffs = target_ages - input_ages

    # 4. 计算一个能代表整个批次的“代表性差值”
    #    我们使用所有样本实际年龄差的【绝对值的平均数】
    #    使用 .float() 是为了精确计算平均值
    representative_diff = torch.mean(torch.abs(actual_age_diffs.float())).item()

    # 5. 基于这个代表性差值，计算一个统一的 aged_strength
    aged_strength = get_age_strength(args, representative_diff, one_threshold=args.one_threshold)
    # print(actual_age_diffs, representative_diff, aged_strength)

    target_face_descriptions = common_utils._generate_prompts(target_ages, batch["gender"])

    original_sizes = batch["original_sizes"]
    crop_top_lefts = batch["crop_top_lefts"]
    target_sizes = torch.tensor([[args.resolution, args.resolution]] * batch_size, device=accelerator.device, dtype=torch.long)
    add_time_ids = torch.cat((original_sizes, crop_top_lefts, target_sizes), dim=1).to(dtype=weight_dtype)

    # 保持 tensor 格式或按需转换为 list
    input_ages_list = [age for age in input_ages]
    target_ages_list = [age for age in target_ages]

    return {
        "input_pixels": input_pixels, 
        "target_face_descriptions": target_face_descriptions, 
        "input_attrs": input_ages_list,
        "target_attrs": target_ages_list, 
        "aged_strength": aged_strength,      # 统一的、基于平均变化计算的 strength
        "age_diffs": actual_age_diffs,       # 返回每个样本的实际年龄差 Tensor，信息更全
        "add_time_ids": add_time_ids
    }


def get_model_input_and_noise(args, image_processor, input_pixels, weight_dtype, vae, device):
    input_pixels = image_processor.preprocess(input_pixels) # convert to: -1~1
    input_pixels = input_pixels.to(device=device)
    
    # Convert images to latent space
    if args.pretrained_vae_model_name_or_path is not None:
        input_pixels = input_pixels.to(dtype=weight_dtype)
    else:
        input_pixels = input_pixels

    model_inputs = vae.encode(input_pixels).latent_dist.sample()
    model_inputs = model_inputs * vae.config.scaling_factor


    if args.pretrained_vae_model_name_or_path is None:
        model_inputs = model_inputs.to(weight_dtype)


    noises = torch.randn_like(model_inputs)
    if args.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noises += args.noise_offset * torch.randn(
            (model_inputs.shape[0], model_inputs.shape[1], 1, 1), device=model_inputs.device
        )

    return model_inputs, noises


def get_arcface_embedding(
    face_project_use_hidden_state,
    input_pixel_list,
    arcFace,
    arcface_model,
    face_project_model,
    weight_dtype,
    age_pixels_list=None,
    additional_age_pixels=None,
    swr_alpha=1.0,
    swr_beta=1.2
):

    batch_size = len(input_pixel_list)

    with torch.no_grad():
        # 处理输入脸图像
        face_inputs = arcFace.preprocess(input_pixel_list)
        face_embedding = arcface_model(face_inputs, return_hidden_state=face_project_use_hidden_state).detach()

        # 如果有 age 相关图像输入
        if age_pixels_list is not None:
            face_inputs_age = arcFace.preprocess(age_pixels_list)
            face_embedding_age = arcface_model(face_inputs_age, return_hidden_state=face_project_use_hidden_state).detach()

            if additional_age_pixels is not None:
                face_inputs_age1 = arcFace.preprocess(additional_age_pixels[0])
                face_embedding_age1 = arcface_model(face_inputs_age1, return_hidden_state=face_project_use_hidden_state).detach()

                face_inputs_age2 = arcFace.preprocess(additional_age_pixels[1])
                face_embedding_age2 = arcface_model(face_inputs_age2, return_hidden_state=face_project_use_hidden_state).detach()

            # 存储处理后的人脸 embedding
            processed_embeddings = []

            for i in range(batch_size):
                embeddings = [face_embedding[i], face_embedding_age[i]]
                if additional_age_pixels is not None:
                    embeddings.extend[face_embedding_age1[i], face_embedding_age2[i]]

                wo_batch = torch.stack(embeddings)  # shape: [N, C]
                wo_batch = punish_wight(
                    wo_batch.T.to(dtype=torch.float32), 
                    wo_batch.size(0), 
                    alpha=swr_alpha, 
                    beta=swr_beta
                ).T.to(dtype=face_embedding.dtype)

                processed_embeddings.append(wo_batch[0])  # 只取处理后的第一个作为主 embedding

                # 显式释放
                del wo_batch, embeddings

            # 合并处理后的 embedding
            face_embedding = torch.stack(processed_embeddings, dim=0)
            del processed_embeddings  # 显式释放

    # 投影阶段（可能用于降维或特征对齐）
    project_face_embedding = face_project_model(face_embedding.to(weight_dtype))
    project_face_embedding = project_face_embedding.unsqueeze(dim=1)  # shape: [B, 1, D]

    return project_face_embedding


def get_aged_image(args, image_processor, input_pixels, aged_strength, models, accelerator, weight_dtype, prompt_embeds, unet_added_conditions):
    model_inputs, noises = get_model_input_and_noise(args, image_processor, input_pixels, weight_dtype, models['vae'], accelerator.device)

    timesteps, latents = get_init_latents(noises, models['noise_scheduler'], model_inputs, args.infer_step, aged_strength)
    
    common_utils.manage_adapters(models['unet'], disable=True)    
    # print(unet_added_conditions['time_ids'].shape)
    with torch.no_grad():
        _, aged_pixels = unet_forward(timesteps, prompt_embeds, image_processor, weight_dtype, models['vae'], models['noise_scheduler'], 
                    unet_added_conditions, models['unet'], latents)
    
    common_utils.manage_adapters(models['unet'], disable=False)  

    return aged_pixels      


def perform_forward_pass_with_ID(args, inputs, accelerator, weight_dtype, image_processor, models):

    # Predict the noise residual
    # 1. get age image
    unet_added_conditions = {"time_ids": inputs['add_time_ids']}
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders=[models['text_encoder_one'], models['text_encoder_two']],
        tokenizers=[models['tokenizer_one'], models['tokenizer_two']],
        prompt=inputs['target_face_descriptions'],
        # text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
    )
        
    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

    aged_pixels = get_aged_image(args, image_processor, inputs["input_pixels"], inputs["aged_strength"], models, accelerator, weight_dtype, prompt_embeds, unet_added_conditions) 
    # 2. improve ID

    # 2.1 get arcface embedding
    embedding_list = []
    if args.use_arcface_project:
        input_pixel_list = [inputs["input_pixels"][i] for i in range(inputs["input_pixels"].shape[0])]
        if args.use_swr:
            if args.use_four_image_swr:
                additional_age_pixels = []
                for strength in args.all_strength:
                    if strength == inputs["aged_strength"]:
                        continue
                    aged_pixels_add = get_aged_image(args, image_processor, inputs["input_pixels"], strength, models, accelerator, weight_dtype, prompt_embeds, unet_added_conditions) 
                    age_pixels_add_list = [aged_pixels_add[i] for i in range(aged_pixels_add.shape[0])]
                    additional_age_pixels.append(age_pixels_add_list)
            else:
                additional_age_pixels = None

            age_pixels_list = [aged_pixels[i] for i in range(aged_pixels.shape[0])]
            project_face_embedding = get_arcface_embedding(args.face_project_use_hidden_state, input_pixel_list, models["arcFace"], models["arcface_model"], 
                                                        models["face_project_model"], weight_dtype, age_pixels_list=age_pixels_list, additional_age_pixels=additional_age_pixels, 
                                                        swr_alpha=args.swr_alpha, swr_beta=args.swr_beta)
        else:
            project_face_embedding = get_arcface_embedding(args.face_project_use_hidden_state, input_pixel_list, models["arcFace"], models["arcface_model"], 
                                                        models["face_project_model"], weight_dtype, age_pixels_list=None)

        embedding_list.append(project_face_embedding)

    # 2.2 get image embedding
    if args.use_clip_project:
        input_pixels_list = [inputs["input_pixels"][i] for i in range(inputs["input_pixels"].shape[0])]
        with torch.no_grad():
            maped_age_clip_embedding = map_attr(input_pixels_list, inputs["input_attrs"], inputs["target_attrs"], models["clip_l_model"], models["clip_l_processor"], accelerator.device, args.clip_map_model)

        clip_face_embedding = models["clip_project_model"](maped_age_clip_embedding.to(weight_dtype))

        clip_face_embedding = clip_face_embedding.unsqueeze(dim=1) 

        embedding_list.append(clip_face_embedding)

    project_face_embedding = torch.cat(embedding_list, dim=1)

    model_inputs, noises = get_model_input_and_noise(args, image_processor, aged_pixels, weight_dtype, models['vae'], accelerator.device)

    # timesteps, latents = get_init_latents_age(noises, models['noise_scheduler'], model_inputs, 4, ages_diff)
    timesteps, latents = get_init_latents(noises, models['noise_scheduler'], model_inputs, 4, args.id_strength)

    noise_pred_init, final_pixels = unet_forward(timesteps, project_face_embedding, image_processor, weight_dtype, models['vae'], models['noise_scheduler'], 
                 unet_added_conditions, models['unet'], latents)
        

    ### Debug
    if args.debug_index == 0 and accelerator.is_main_process:
        idx = 0
        # 生成三张待拼接的图片
        output_img = common_utils.img_to_pil(final_pixels, image_processor)[idx]
        aged_img = common_utils.img_to_pil(aged_pixels, image_processor)[idx]
        input_img = common_utils.img_to_pil(inputs['input_pixels'], image_processor)[idx]
        
        # 计算拼接尺寸
        combined = common_utils.horizontal_concat([input_img, aged_img, output_img])
        
        # 保存拼接结果
        combined.save("temp/combined.png")
        input_img.save("temp/input_img.png")
        aged_img.save("temp/aged_img.png")
        output_img.save("temp/output_img.png")
        
        # 调试输出
        print(inputs['target_face_descriptions'][idx])

        pass
    ###

    return noise_pred_init, noises, model_inputs, timesteps, final_pixels, aged_pixels