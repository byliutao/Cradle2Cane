import os
import glob
import argparse
from PIL import Image
import torch
from typing import List
from lib.arcface import get_model
import torch.nn.functional as F
from torchvision import transforms

class ArcFaceWrapper():
    def __init__(self, weight, network, device, weight_dtype=torch.float32, input_size=112):
        use_fp16 = weight_dtype == torch.float16
        self.net = get_model(network, fp16=use_fp16).to(device)
        self.net.load_state_dict(torch.load(weight, map_location=device))
        self.net.eval()
        self.net.requires_grad_(False)

        self.device = device
        self.input_size = input_size
        self.weight_dtype = weight_dtype

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def preprocess(self, images: List):
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
            images = F.interpolate(images, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        else:
            image_list = []
            for image in images:
                image = image.convert('RGB')
                image_list.append(self.transform(image))
            images = torch.stack(image_list)

        images = self.normalize(images)
        images = images.to(self.device, dtype=self.weight_dtype)
        return images

    def __call__(self, images: List):
        imgs = self.preprocess(images)
        with torch.no_grad():
            return self.net(imgs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', type=str, required=True, help="Reference image folder")
    parser.add_argument('--folder2', type=str, required=True, help="Folder containing subfolders to compare")
    parser.add_argument('--network', type=str, default='r100')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    arcface = ArcFaceWrapper(args.weight, args.network, args.device)

    # 找到 folder1 中的图像
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    ref_images = []
    for ext in image_extensions:
        ref_images.extend(glob.glob(os.path.join(args.folder1, ext)))
    if not ref_images:
        print(f"No reference images found in {args.folder1}")
        return

    # 遍历 folder2 中的所有子目录
    subdirs = [os.path.join(args.folder2, d) for d in os.listdir(args.folder2) if os.path.isdir(os.path.join(args.folder2, d))]
    if not subdirs:
        print(f"No subdirectories found in {args.folder2}")
        return

    total_similarity = 0.0
    total_pairs = 0

    for subdir in subdirs:
        if int(os.path.basename(subdir)) >= 80 or int(os.path.basename(subdir)) <= 5:
            continue
        sim_sum = 0.0
        sim_count = 0
        print(f"\nProcessing subdirectory: {os.path.basename(subdir)}")

        for ref_path in ref_images:
            fname = os.path.basename(ref_path)
            cmp_path = os.path.join(subdir, fname)
            if not os.path.exists(cmp_path):
                continue

            try:
                img1 = Image.open(ref_path)
                img2 = Image.open(cmp_path)
                feat1 = arcface([img1])
                feat2 = arcface([img2])
                sim = F.cosine_similarity(feat1, feat2).item()
            except Exception as e:
                print(f"Error comparing {fname}: {e}")
                continue

            sim_sum += sim
            sim_count += 1
            total_similarity += sim
            total_pairs += 1

        if sim_count > 0:
            avg_sim = sim_sum / sim_count
            print(f"Subdir {os.path.basename(subdir)} average ArcFace similarity: {avg_sim:.4f}")
        else:
            print(f"No matching images in {os.path.basename(subdir)}")

    if total_pairs > 0:
        overall_avg = total_similarity / total_pairs
        print(f"\n==============================")
        print(f"Processed total {total_pairs} image pairs")
        print(f"Overall average ArcFace similarity: {overall_avg:.4f}")
        print(f"==============================")
    else:
        print("No image pairs compared.")

if __name__ == "__main__":
    main()


# python -m lib.eval.arcface_eval --folder1 models/cele_plus_200 --folder2 temp/cele_plus_200/cradle_2  --weight models/backbone.pth  --network r100