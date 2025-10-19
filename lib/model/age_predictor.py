import argparse
from typing import List
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from lib.mivolo.model.mi_volo import MiVOLO


class AgePredictor:
    def __init__(self, model_path, device, weight_dtype, input_size=224, eval=True, require_grad=False):
        self.mivolo_model = MiVOLO(
            model_path,
            device,
            half=(False if weight_dtype == torch.float32 else True),
            verbose=True,
            use_persons=False
        )
        if eval:
            self.mivolo_model.model.eval()
        self.mivolo_model.model.requires_grad_ = require_grad

        self.device = device
        self.input_size = input_size
        self.weight_dtype = weight_dtype

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def preprocess(self, images):
        """
        Accepts:
            - list of PIL.Image
            - list of tensors [C,H,W] in 0~1
            - tensor [B,C,H,W] in 0~1
        Returns:
            - tensor [B,C,input_size,input_size] on self.device with dtype self.weight_dtype
        """
        # Case 1: already a batch tensor
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:  # [B,C,H,W]
                imgs = images
            elif images.ndim == 3:  # single image [C,H,W]
                imgs = images.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected tensor shape: {images.shape}")
            imgs = torch.nn.functional.interpolate(
                imgs,
                size=self.input_size,
                mode='bilinear',
                align_corners=False
            )

        # Case 2: list of tensors or PIL
        elif isinstance(images, list):
            if isinstance(images[0], torch.Tensor):
                imgs = torch.stack(images)  # [B,C,H,W]
                imgs = torch.nn.functional.interpolate(
                    imgs,
                    size=self.input_size,
                    mode='bilinear',
                    align_corners=False
                )
            elif isinstance(images[0], Image.Image):
                img_list = [self.transform(img) for img in images]
                imgs = torch.stack(img_list)
            else:
                raise ValueError("List elements should be PIL.Image or torch.Tensor")
        else:
            raise ValueError("images should be torch.Tensor or list")

        if imgs.max() > 1.0 or imgs.min() < 0.0:
            raise ValueError("image tensor values should be in [0,1]")

        imgs = self.normalize(imgs).to(self.device, dtype=self.weight_dtype)
        return imgs

    def __call__(self, images, return_feature=False):
        """
        images: list of PIL.Image, list of [C,H,W] tensor, or [B,C,H,W] tensor
        return_feature: bool, whether to return the feature vector instead of age scalar
        """
        imgs = self.preprocess(images)
        features_or_age = self.mivolo_model.inference(imgs, return_feature=return_feature)

        if return_feature:
            return features_or_age  # [B, feature_dim] if batch input

        # convert feature to age scalar
        age = features_or_age * (self.mivolo_model.meta.max_age - self.mivolo_model.meta.min_age) + self.mivolo_model.meta.avg_age
        return age


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Validation")
    parser.add_argument("--checkpoint", default="/data/model/model_only_age_imdb_4.29.pth.tar", type=str, help="path to mivolo checkpoint")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    parser.add_argument("--image_path1", default="/new-common-data/tao.liu/dataset/ffhq_crop_gray/40.0_female_03758.png", type=str)
    parser.add_argument("--image_path2", default="/new-common-data/tao.liu/dataset/ffhq_crop_gray/1.0_female_05680.png", type=str)
    parser.add_argument("--half", action="store_true", default=False, help="use half-precision model")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    totensor = transforms.ToTensor()

    agePredictor = AgePredictor(args.checkpoint, args.device, torch.float32)

    # 单张预测
    age1 = agePredictor(totensor(Image.open(args.image_path1)).unsqueeze(0))
    age2 = agePredictor(totensor(Image.open(args.image_path2)).unsqueeze(0))
    print("Predicted ages:", age1, age2)

    # 批量预测 feature
    batch_tensor = torch.stack([totensor(Image.open(p)) for p in [args.image_path1, args.image_path2]])
    features = agePredictor(batch_tensor, return_feature=True)
    age_cosine = F.cosine_similarity(features[0:1], features[1:2]).mean()
    print("Cosine similarity of features:", age_cosine)
