import os
import shutil

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
)
from diffusers.image_processor import VaeImageProcessor


from lib.utils import common_utils, train_utils, config_utils
from lib.model.gan_loss import generator_hinge_loss, discriminator_hinge_loss



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)


def print_mem(tag=""):
    print(f"[{tag}] Allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB, "
          f"Reserved: {torch.cuda.memory_reserved()/1e6:.1f} MB")

def main(args):
    accelerator, repo_id = train_utils.prepare_training_environment(args, logger)
    
    models, train_models_name_G, weight_dtype = train_utils.load_all_models(args, accelerator, logger)

    # gan loss
    if args.use_gan_loss:
        optimizer_D = torch.optim.Adam(models["D"].parameters(), lr=1e-4, betas=(0.5, 0.999))
        models["D"], optimizer_D = accelerator.prepare(models["D"], optimizer_D) # 多卡同步D


    train_utils.setup_save_and_load_hooks(args, accelerator, models, logger)

    if args.gradient_checkpointing:
        models["unet"].enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        cast_models = []        
        for model_name in train_models_name_G:
            cast_models.append(models[model_name])
        cast_training_params(cast_models, dtype=torch.float32)

    # 准备数据集和数据加载器
    train_dataset, train_dataloader = train_utils.prepare_dataset_and_dataloader(args, accelerator)
    
    # 计算训练步数
    num_update_steps_per_epoch, overrode_max_train_steps = train_utils.calculate_training_steps(args, train_dataloader)

    # 准备优化器和学习率调度器
    optimizer, params_to_optimize, lr_scheduler = train_utils.prepare_optimizer_and_scheduler(
        args, models, train_models_name_G, accelerator
    )

    models, optimizer, train_dataloader, lr_scheduler, num_update_steps_per_epoch = train_utils.prepare_for_training(
            args, accelerator, models, train_models_name_G, optimizer,
            train_dataloader, lr_scheduler, num_update_steps_per_epoch, overrode_max_train_steps
        )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Output dir = {args.output_dir}")


    # 从检查点恢复(如果需要)
    global_step, first_epoch, initial_global_step = train_utils.resume_from_checkpoint(args, accelerator, num_update_steps_per_epoch)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    vae_scale_factor = 2 ** (len(models['vae'].config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)


    for epoch in range(first_epoch, args.num_train_epochs):
        for model_name in train_models_name_G:
            models[model_name].train()
        
        if args.use_gan_loss:
            models['D'].train()

        train_loss_g = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(models[name] for name in train_models_name_G):
                all_loss_G = {}

                if args.use_avg:
                    inputs = train_utils.get_inputs_from_batch_ffhq_avg(args, batch, accelerator, weight_dtype)                       
                else:
                    inputs = train_utils.get_inputs_from_batch_ffhq(args, batch, accelerator, weight_dtype)                       
                
                noise_pred, noises, model_inputs, timesteps, output_pixels, aged_pixels = train_utils.perform_forward_pass_with_ID(args, inputs, accelerator, weight_dtype, image_processor, models)
                
                del noise_pred, noises, model_inputs, timesteps
                torch.cuda.empty_cache()

                output_pixels = output_pixels.to(dtype=torch.float32)
                input_pixels = inputs["input_pixels"]


                # diffusion_loss = common_utils.diffusion_loss(all_loss_G, args, models['noise_scheduler'], noises, noise_pred, model_inputs, timesteps, args.diffusion_loss_weight)
                age_loss = common_utils.age_loss(all_loss_G, output_pixels, inputs["target_attrs"], models['agePredictor'], weight=args.age_loss_weight)
                age_loss_2 = common_utils.age_loss_2(all_loss_G, output_pixels, aged_pixels, models['agePredictor'], weight=args.age_loss_2_weight)
                id_cos_loss = common_utils.arcface_loss(all_loss_G, models['arcFace'], input_pixels, output_pixels, inputs["aged_strength"], args.id_cos_loss_weight,)                
                pixel_mse_loss = common_utils.image_mse_loss(all_loss_G, input_pixels, output_pixels, args.pixel_mse_loss_use_weight, args.pixel_mse_loss_weight)
                lpips_loss = common_utils.lpips_loss(all_loss_G, models['loss_fn_alex'], input_pixels, output_pixels, args.lpips_loss_use_weight, args.lpips_loss_weight)
                ssim_loss = common_utils.ssim_loss(all_loss_G, input_pixels, output_pixels, args.ssim_loss_weight)


                del aged_pixels
                torch.cuda.empty_cache()

                if args.use_gan_loss:
                    # gan_weight = args.start_g_loss_weigth * (1.0 - global_step/args.max_train_steps) + args.end_g_loss_weigth * global_step/args.max_train_steps
                    g_loss = generator_hinge_loss(all_loss_G, models["D"], output_pixels, args.g_loss_weight)


                # print("id_cos:",args.id_cos_loss_weight, "age1:", args.age_loss_weight, "age2:", args.age_loss_2_weight, 
                #       "pixel:", args.pixel_mse_loss_weight, "lpips:", args.lpips_loss_weight, "ssim:", args.ssim_loss_weight, 
                #       "gan_loss:", args.g_loss_weight if args.use_gan_loss else 0.0)

                # Sum all the losses
                g_loss = 0.0
                for loss in all_loss_G:
                    g_loss += all_loss_G[loss]     
                    
                if args.debug_loss and "filenames" in batch:
                    for fname in batch["filenames"]:
                        accelerator.log({"loss_for_" + fname: g_loss}, step=global_step)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(g_loss.repeat(args.train_batch_size)).mean()
                train_loss_g += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(g_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if args.use_gan_loss and global_step % args.d_loss_update_step == 0:
                # 判别器训练阶段使用 detach()，防止第二次反向传播
                with accelerator.accumulate(models["D"]):
                    # fake_detached = [img.detach() for img in output_pixels]
                    d_loss = discriminator_hinge_loss(models["D"], input_pixels, output_pixels.detach(), args.d_loss_weight)
                    accelerator.backward(d_loss)
                    
                    optimizer_D.step()
                    optimizer_D.zero_grad()

            
            del output_pixels
            torch.cuda.empty_cache()
            


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss_g": train_loss_g}, step=global_step)

                for loss in all_loss_G:
                    # has_grad = all_loss_G[loss].requires_grad and all_loss_G[loss].grad_fn is not None
                    accelerator.log({f"{loss}": all_loss_G[loss].item()}, step=global_step)
                
                if args.use_gan_loss and global_step % args.d_loss_update_step == 0:
                    accelerator.log({"d_loss": d_loss.item()}, step=global_step)


                train_loss_g = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        filtered_config = {k: v for k, v in vars(args).items() if not isinstance(v, list)}                            
                        import yaml
                        with open(f'{save_path}/hparams.yml', 'w') as f:
                            yaml.dump(filtered_config, f, sort_keys=False, allow_unicode=True)

                        logger.info(f"Saved state to {save_path}")

                    if args.validation_image is not None and global_step % args.validation_steps == 0:
                        train_utils.perform_validation(args, accelerator, logger, global_step, models, weight_dtype)


            # Log individual losses
            logs = {
                "step_loss": g_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            for loss in all_loss_G:
                logs[loss] = all_loss_G[loss].detach().item()
            
            if args.use_gan_loss:
                logs["d_loss"] = d_loss.detach().item()
                
            
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break


    # 保存模型权重
    train_utils.save_model_weights(args, accelerator, models, train_models_name_G)
    
    # 执行最终推理
    if accelerator.is_main_process:
        train_utils.perform_final_inference(args, accelerator, models, weight_dtype, repo_id)

    accelerator.end_training()


if __name__ == "__main__":
    args = config_utils.parse_args()
    main(args)