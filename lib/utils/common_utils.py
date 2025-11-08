import torch
import torch.nn.functional as F
from diffusers.training_utils import cast_training_params, compute_snr
from PIL import Image
import pytorch_msssim
from pathlib import Path
import os


def generate_prompts(target_ages, labels):
    age_list = target_ages
    gender_list = [labels["gender"]]
    prompts = []

    for age, gender in zip(age_list, gender_list):
        prompt = f"a face image of a {age} years old {gender}"
        prompts.append(prompt)

    return prompts


def _generate_prompts(target_ages, gender_list):
    prompts = []
    for age, gender in zip(target_ages, gender_list):
        prompt = f"a face image of a {age} years old {gender}"
        prompts.append(prompt)

    return prompts


def get_labels_from_path(path_name):

    path_obj = Path(path_name)
    path_without_extension = str(path_obj.with_suffix(''))

    labels = os.path.basename(path_without_extension).split("_")
    if len(labels) == 2:
        return {"age": int(float(labels[0])), 
            "gender": labels[1]}
    elif len(labels) == 3:
        return {"age": int(float(labels[0])), 
            "gender": labels[1],
            "index": labels[2]}
    elif len(labels) == 4:
        return {"age": int(float(labels[0])), 
            "gender": labels[1],
            "index": labels[2],
            "race": labels[3]}
                
    else:
        raise ValueError(f"Invalid file name: {path_name}")



def load_and_process_image(image_path, target_size=512):
    """
    Load an image, crop it to a centered square, and resize it to target_size x target_size.
    
    Parameters:
    - image_path (str): Path to the image file
    - target_size (int): Size of the target square image (default: 512)
    
    Returns:
    - PIL.Image: Processed square image
    """
    # Load the image
    input_image = Image.open(image_path).convert('RGB')
    
    # Get the dimensions
    width, height = input_image.size
    
    # Find the shortest side
    min_dim = min(width, height)
    
    # Calculate crop box (left, upper, right, lower)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    # Crop the image to a square
    input_image = input_image.crop((left, top, right, bottom))
    
    # Resize to target_size x target_size
    input_image = input_image.resize((target_size, target_size), Image.LANCZOS)
    
    return input_image


def img_to_pil(images, image_processor): # value show be at (0~1)
    images = (images * 255).round()
    images = image_processor.pt_to_numpy(images.clone().detach()).astype("uint8")
    pil_images = [Image.fromarray(image[:, :, :3]) for image in images]
    
    return pil_images


def horizontal_concat(images, align="top", bg_color=(255, 255, 255)):
    """
    横向拼接PIL图像列表
    
    参数：
    images: list of PIL.Image - 待拼接图像列表（至少包含1张图像）
    align: str - 垂直对齐方式，可选 top/center/bottom（默认top）
    bg_color: tuple - 背景颜色 RGB 元组（默认纯白）
    
    返回：
    PIL.Image 拼接后的新图像
    """
    if not images:
        raise ValueError("Empty image list")
    if any(not isinstance(img, Image.Image) for img in images):
        raise TypeError("All elements must be PIL.Image objects")

    # 计算画布尺寸
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    # 创建背景画布
    combined = Image.new('RGB', (total_width, max_height), color=bg_color)
    
    # 设置粘贴坐标
    x_offset = 0
    for img in images:
        # 计算垂直偏移
        if align == "center":
            y_offset = (max_height - img.height) // 2
        elif align == "bottom":
            y_offset = max_height - img.height
        else:  # top
            y_offset = 0
        
        combined.paste(img, (x_offset, y_offset))
        x_offset += img.width
    
    return combined


def manage_adapters(model, disable=True):
    if disable:
        if hasattr(model, 'module'):
            model.module.disable_adapters()
        else:
            model.disable_adapters()
    else:
        if hasattr(model, 'module'):
            model.module.enable_adapters()
        else:
            model.enable_adapters()


def create_weight_mask(height, width, center_weight=2.0, edge_weight=0.5, sigma=0.5):
    """
    生成一个权重掩码，使得中心区域的权重较高，边缘区域的权重较低。
    使用高斯分布来平滑权重过渡。
    """
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
    distance = torch.sqrt(x**2 + y**2)  # 计算每个像素点到中心的距离
    weight_mask = torch.exp(-distance**2 / (2 * sigma**2))  # 高斯权重
    weight_mask = weight_mask * (center_weight - edge_weight) + edge_weight  # 归一化调整权重范围
    return weight_mask


def image_mse_loss(all_loss, input_pixels, output_pixels, use_weight_map=True, weight=1.0):
    if weight == 0:
        return -1     


    # 假设 input_pixels 和 output_pixels 的尺寸是 (B, C, H, W)
    B, C, H, W = input_pixels.shape

    # 生成权重掩码，并扩展维度以匹配图像
    if use_weight_map:
        weight_mask = create_weight_mask(H, W).to(input_pixels.device)  # (H, W)
        weight_mask = weight_mask.unsqueeze(0).unsqueeze(0)  # 变成 (1, 1, H, W)，用于广播

        # 计算加权 MSE Loss
        pixel_mse_loss = (F.mse_loss(input_pixels, output_pixels, reduction="none") * weight_mask).mean()
    else:
        # 计算加权 MSE Loss
        pixel_mse_loss = F.mse_loss(input_pixels, output_pixels, reduction="none").mean()

    # if age_diff != 0:
    #     pixel_mse_loss *= 0

    pixel_mse_loss *= weight

    if weight > 0:
        all_loss["pixel_mse_loss"] = pixel_mse_loss

    return pixel_mse_loss



def lpips_loss(all_loss, loss_fn_alex, input_pixels, output_pixels, use_weight_map=False, weight=1.0):
    if weight == 0:
        return -1     

    # 假设 input_pixels 和 output_pixels 的尺寸是 (B, C, H, W)
    B, C, H, W = input_pixels.shape

    # 生成权重掩码，并扩展维度以匹配图像
    if use_weight_map:
        weight_mask = create_weight_mask(H, W).to(input_pixels.device)  # (H, W)
        weight_mask = weight_mask.unsqueeze(0).unsqueeze(0)  # 变成 (1, 1, H, W)，用于广播

        input_pixels_weighted = input_pixels * weight_mask
        output_pixels_weighted = output_pixels * weight_mask

        # **计算 LPIPS Loss**
        lpips_loss = loss_fn_alex.forward(input_pixels_weighted, output_pixels_weighted, normalize=True)
    else:
        lpips_loss = loss_fn_alex.forward(input_pixels, output_pixels, normalize=True)
    
    lpips_loss = lpips_loss.mean()

    lpips_loss *= weight

    if weight > 0:
        all_loss["lpips_loss"] = lpips_loss

    return lpips_loss



def arcface_loss(all_loss, arcFace, input_pixels_list, output_pixels_list, aged_strength, weight=1.0):
    if weight == 0:
        return -1     
    input_embeds = arcFace(input_pixels_list)
    output_embeds = arcFace(output_pixels_list)


    id_cos_loss = 1 - F.cosine_similarity(input_embeds, output_embeds).mean()
    # id_cos_loss *= 1 - aged_strength

    id_cos_loss *= weight

    if weight > 0:
        all_loss["id_cos_loss"] = id_cos_loss

    return id_cos_loss


# def age_loss(all_loss, output_pixels, target_ages, agePredictor, weight=1.0):
#     if weight == 0:
#         return -1     
#     target_ages_tensor = torch.stack(target_ages).to(torch.float32) / 100.0 # normalize to 0~1
#     output_ages_tensor = agePredictor(output_pixels).squeeze(dim=1) / 100.0
    
#     age_loss = F.mse_loss(output_ages_tensor, target_ages_tensor, reduction="mean") * 10
    
#     age_loss *= weight

#     if weight > 0:
#         all_loss["age_loss"] = age_loss

#     return age_loss

def age_loss(all_loss, output_pixels_list, target_ages, agePredictor, weight=1.0, beta=1.0):
    if weight == 0:
        return -1     

    # 年龄归一化
    target_ages_tensor = torch.stack(target_ages).to(torch.float32) / 100.0
    output_ages_tensor = agePredictor(output_pixels_list).squeeze(dim=1) / 100.0

    # 使用 Smooth L1 Loss
    age_loss = F.smooth_l1_loss(output_ages_tensor, target_ages_tensor, reduction="mean", beta=beta) * 10

    age_loss *= weight

    if weight > 0:
        all_loss["age_loss"] = age_loss

    return age_loss


def age_loss_2(all_loss, output_pixels_list, age_pixels_list, agePredictor, mode='none', weight=1.0):
    if weight == 0:
        return -1     
    output_feature = agePredictor(output_pixels_list, return_feature=True)
    age_feature = agePredictor(age_pixels_list, return_feature=True)

    if mode == 'avg':
        output_feature = output_feature.mean(dim=2)
        age_feature = age_feature.mean(dim=2)
    elif mode == "token":
        output_feature = output_feature[:,:,0]
        age_feature = age_feature[:,:,0]

    age_loss = 1 - F.cosine_similarity(output_feature, age_feature).mean()
    # age_loss = F.mse_loss(output_feature, age_feature, reduction="mean")

    age_loss *= weight

    if weight > 0:
        all_loss["age_loss_2"] = age_loss

    return age_loss


def diffusion_loss(all_loss, args, noise_scheduler, noises, noise_pred, model_inputs, timesteps, weight=1.0):
    if weight == 0:
        return -1 
    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noises
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(model_inputs, noises, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    if args.snr_gamma is None:
        diffusion_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        diffusion_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        diffusion_loss = diffusion_loss.mean(dim=list(range(1, len(diffusion_loss.shape)))) * mse_loss_weights
        diffusion_loss = diffusion_loss.mean()

    diffusion_loss *= weight

    if weight > 0:
        all_loss["diffusion_loss"] = diffusion_loss
    
    return diffusion_loss 


def ssim_loss(all_loss, input_pixels, output_pixels, weight=1.0):
    if weight == 0:
        return -1 
    ms_ssim_loss = 1 - pytorch_msssim.ms_ssim(output_pixels, input_pixels, data_range=1.0, size_average=True)


    ms_ssim_loss *= weight

    if weight > 0:
        all_loss["ms_ssim_loss"] = ms_ssim_loss

    return ms_ssim_loss


def stitch_images(images, images_per_row=10):
    """
    将一组图像拼接成一个网格图像。

    Args:
        images (list of PIL.Image): 要拼接的图像列表。
        images_per_row (int): 每行拼接的图像数量（默认是10）。

    Returns:
        PIL.Image: 拼接后的图像。
    """
    num_images = len(images)
    rows = (num_images + images_per_row - 1) // images_per_row  # 向上取整

    first_image = images[0]
    width, height = first_image.size

    stitched_image = Image.new('RGB', (width * images_per_row, height * rows))

    for i, img in enumerate(images):
        row = i // images_per_row
        col = i % images_per_row
        stitched_image.paste(img, (col * width, row * height))

    return stitched_image