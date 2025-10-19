import torch
from PIL import Image
from transformers.models.clip.modeling_clip import _get_vector_norm 


def normalize(features):
    return features / _get_vector_norm(features)


def slerp(val, low, high):
    """球面线性插值"""
    low_norm = low / low.norm(dim=-1, keepdim=True)
    high_norm = high / high.norm(dim=-1, keepdim=True)
    dot = (low_norm * high_norm).sum(-1, keepdim=True)
    omega = torch.acos(torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7))
    sin_omega = torch.sin(omega)
    factor1 = torch.sin((1.0 - val) * omega) / sin_omega
    factor2 = torch.sin(val * omega) / sin_omega
    return factor1 * low_norm + factor2 * high_norm


def map_attr_single(image, source_attr, target_attr, clip_model, clip_processor, method="vector", device="cuda"):
    """
    处理单张图像的年龄特征映射。
    支持 vector 或 angle 差异方式。
    """
    # 预处理图像 & 文本
    clip_processor.image_processor._valid_processor_keys = {} # TODO: bug
    image_tensor = clip_processor(images=image, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)
    text_inputs = clip_processor(
        text=[f"{source_attr} years old", f"{target_attr} years old"],
        padding=True,
        return_tensors="pt"
    ).to(device)


    # 特征提取并归一化
    image_feat = normalize(clip_model.get_image_features(pixel_values=image_tensor))

    if source_attr == target_attr:
        return image_feat, image_feat, image_feat
    
    text_feat = normalize(clip_model.get_text_features(**text_inputs))

    if method == "vector":
        age_shift = (text_feat[1] - text_feat[0].unsqueeze(0))
    elif method == "angle":
        mid_text = slerp(0.5, text_feat[1], text_feat[0])
        age_shift = mid_text - text_feat[0].unsqueeze(0)
    else:
        raise ValueError("method must be 'vector' or 'angle'.")

    modified_feat = normalize(image_feat + age_shift)
    return image_feat, modified_feat, text_feat


def map_attr(image_inputs, source_attrs, target_attrs, clip_model, clip_processor, device, method="vector"):
    """
    将输入图像的年龄信息映射到对应的目标年龄（支持批量处理）。
    
    参数:
        image_inputs (list[str | PIL.Image.Image | torch.Tensor]): 输入图像列表。
        source_ages (list[int]): 每张图像的原始年龄。
        target_ages (list[int]): 每张图像的目标年龄。
        clip_model (CLIPModel): 预训练的 CLIP 模型。
        clip_processor (CLIPProcessor): CLIP 处理器。
        device (str): 计算设备。
        method (str): "vector" 直接相减，"angle" 使用角度差/slerp方式。

    返回:
        torch.Tensor: 映射后的图像特征 (batch_size, feature_dim)。
    """

    if not isinstance(image_inputs, list):
        raise ValueError("image_inputs 必须是列表，每张图像可以是路径、PIL 或 Tensor。")
    if not isinstance(source_attrs, list) or not isinstance(target_attrs, list):
        raise ValueError("source_ages 和 target_ages 必须是列表。")
    if len(image_inputs) != len(source_attrs) or len(image_inputs) != len(target_attrs):
        raise ValueError("image_inputs、source_ages 和 target_ages 的长度必须相同。")


    # 处理图像
    processed_images = []
    for image_input in image_inputs:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, torch.Tensor):
            image = image_input
        else:
            raise ValueError("image_inputs 的元素类型必须为 str、PIL.Image 或 torch.Tensor。")
        processed_images.append(image)


    # 构造文本输入
    mapped_image_features = []
    for image, source_attr, target_attr in zip(processed_images, source_attrs, target_attrs):
        _, mapped_image_feature, _ = map_attr_single(image, source_attr, target_attr, clip_model, clip_processor, method, device)
        mapped_image_features.append(mapped_image_feature.squeeze(0))

    mapped_image_features = torch.stack(mapped_image_features)
    return mapped_image_features  # (batch, dim)


def map_age1(image_inputs, source_ages, target_ages, clip_model, clip_processor, device="cuda"):
    """
    将输入图像的年龄信息映射到对应的目标年龄（支持批量处理，每张图像的年龄不同）。
    
    参数:
        image_inputs (list[str | PIL.Image.Image | torch.Tensor]): 输入图像列表，可混合路径、PIL、Tensor。
        source_ages (list[int]): 每张图像的原始年龄（如 [50, 60, 40]）。
        target_ages (list[int]): 每张图像的目标年龄（如 [20, 30, 25]）。
        clip_model (CLIPModel): 预训练的 CLIP 模型。
        clip_processor (CLIPProcessor): CLIP 处理器。
        device (str): 计算设备 (默认 "cuda")

    返回:
        torch.Tensor: 经过年龄转换的图像特征向量 (batch_size, feature_dim)。
    """
    
    # 统一 batch 维度
    if not isinstance(image_inputs, list):
        raise ValueError("image_inputs 必须是列表，每张图像可以是路径、PIL 或 Tensor。")
    if not isinstance(source_ages, list) or not isinstance(target_ages, list):
        raise ValueError("source_ages 和 target_ages 必须是列表，每张图像对应一个年龄。")
    if len(image_inputs) != len(source_ages) or len(image_inputs) != len(target_ages):
        raise ValueError("image_inputs、source_ages 和 target_ages 的长度必须相同。")

    batch_size = len(image_inputs)

    # 处理不同类型的输入图像
    processed_images = []
    for image_input in image_inputs:
        if isinstance(image_input, str):  # 图像路径
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):  # PIL Image
            image = image_input.convert("RGB")
        elif isinstance(image_input, torch.Tensor):  # 已预处理的 Tensor
            image = image_input
        else:
            raise ValueError("image_inputs 内部元素必须是文件路径 (str)、PIL 图像 (Image.Image) 或 Tensor")

        # 预处理图像
        image_tensor = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        processed_images.append(image_tensor)

    # 合并 Batch 维度
    image_tensors = torch.cat(processed_images, dim=0)  # (batch_size, 3, H, W)

    # 构建文本输入
    text_prompts = [f"{age} years old" for age in source_ages + target_ages]
    text_inputs = clip_processor(
        text=text_prompts,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 提取图像特征
    image_features = clip_model.get_image_features(pixel_values=image_tensors)  # (batch_size, feature_dim)

    # 提取文本特征
    text_features = clip_model.get_text_features(**text_inputs)  # (2 * batch_size, feature_dim)

    # 计算年龄特征偏移
    source_features = text_features[:batch_size]  # (batch_size, feature_dim)
    target_features = text_features[batch_size:]  # (batch_size, feature_dim)
    age_shifts = (source_features - target_features)  # (batch_size, feature_dim)

    # 应用年龄映射
    mapped_image_features = image_features + age_shifts  # (batch_size, feature_dim)

    # L2 归一化
    def normalize(features):
        return features / _get_vector_norm(features)

    return normalize(mapped_image_features)  # (batch_size, feature_dim)
