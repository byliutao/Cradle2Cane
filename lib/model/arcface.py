import argparse
import torch
import torch.nn.functional as F
from typing import List
from PIL import Image
from torchvision import transforms

from lib.arcface import get_model


class ArcFace():
    def __init__(self, weight, network, device, weight_dtype, input_size=112, eval=True, require_grad=False):
        if weight_dtype is torch.float32:
            use_fp16 = False
        else:
            use_fp16 = True

        self.net = get_model(network, fp16=use_fp16).to(device)
        self.net.load_state_dict(torch.load(weight, map_location=device))
        
        self.device = device
        self.input_size = input_size
        self.weight_dtype = weight_dtype

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

        if eval:
            self.net.eval()
        self.net.requires_grad_ = require_grad

    def preprocess(self, images):
        """
        images: List[PIL.Image] | List[Tensor] | Tensor[batch, C, H, W]
        """
        # 如果是单个 Tensor，保证是 4D
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            if images.max() > 1.0 or images.min() < 0.0:
                raise ValueError("Tensor input should be in range 0~1")
            images = F.interpolate(images, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        # 如果是 list
        elif isinstance(images, list):
            if isinstance(images[0], torch.Tensor):
                images = torch.stack(images)
                if images.max() > 1.0 or images.min() < 0.0:
                    raise ValueError("Tensor list input should be in range 0~1")
                images = F.interpolate(images, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
            elif isinstance(images[0], Image.Image):
                image_list = []
                for image in images:
                    image = image.convert('RGB')
                    image_list.append(self.transform(image))
                images = torch.stack(image_list)
            else:
                raise TypeError(f"Unsupported list element type: {type(images[0])}")
        else:
            raise TypeError(f"Unsupported input type: {type(images)}")

        images = self.normalize(images)
        images = images.to(self.device, dtype=self.weight_dtype)
        return images

    def __call__(self, imgs, return_hidden_state=False):
        """
        imgs: List[PIL.Image] | List[Tensor] | Tensor[batch, C, H, W], values in 0~1
        return_hidden_state: 是否返回隐藏状态
        """
        imgs = self.preprocess(imgs)
        feat = self.net(imgs, return_hidden_state=return_hidden_state)
        return feat

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Similarity with ArcFace')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default="/data/model/ms1mv3_arcface_r100_fp16.pth")
    parser.add_argument('--img1', type=str, default="/data/workspace/Aging/others/10002_GoldieHawn_24_f.jpg", help='Path to first face image')
    parser.add_argument('--img2', type=str, default="/data/workspace/Aging/others/10013_GoldieHawn_44_f.jpg", help='Path to second face image')
    args = parser.parse_args()

    arcFace = ArcFace(args.weight, args.network, "cuda", torch.float32)

    input_embeds = arcFace([Image.open(args.img1)])
    output_embeds = arcFace([Image.open(args.img2)])
    cosine_similarity = torch.nn.functional.cosine_similarity(input_embeds, output_embeds)
    print(f"{torch.mean(input_embeds)} {torch.mean(output_embeds)} {cosine_similarity}")

    import torchvision
    transform = torchvision.transforms.ToTensor()
    tensor_input = arcFace([transform(Image.open(args.img1).convert('RGB'))])
    tensor_output = arcFace([transform(Image.open(args.img2).convert('RGB'))])
    cosine_similarity = torch.nn.functional.cosine_similarity(tensor_input, tensor_output)


    print(f"{torch.mean(tensor_input)} {torch.mean(tensor_output)} {cosine_similarity}")
