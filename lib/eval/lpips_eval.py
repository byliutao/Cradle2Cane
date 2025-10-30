import os
import argparse
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# --- Paste your LPIPS related classes and functions here ---
# from lpips import LPIPS, ScalingLayer, NetLinLayer, pn, upsample, spatial_average # Original imports
# from lpips.pretrained_networks import alexnet, vgg16 # Original imports
from torchvision import models as tv
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.vgg import VGG16_Weights

from lpips import LPIPS, ScalingLayer, NetLinLayer, pn, upsample, spatial_average
from lpips.pretrained_networks import alexnet, vgg16
from torchvision import models as tv
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.vgg import VGG16_Weights


# 设置 TORCH_HUB 的值（例如指向自定义模型缓存目录）
# os.environ["TORCH_HOME"] = "/data/model"


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    if isinstance(image, torch.Tensor):
        # 输入为张量时的处理逻辑
        if image.dim() == 3:  # 若维度为[H,W,C]
            tensor = image.permute(2, 0, 1).unsqueeze(0)  # 转换为[N,C,H,W]
        else:
            tensor = image.unsqueeze(0) if image.dim() == 2 else image
        return (tensor / factor - cent).float()
    else:
        # 原处理逻辑（支持numpy数组）
        return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


class fix_alexnet(alexnet):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(weights=AlexNet_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                

class fix_vgg16(vgg16):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


class FIX_LPIPS(LPIPS):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, 
        pnet_rand=False, pnet_tune=False, use_dropout=False, model_path=None, eval_mode=True, verbose=True):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1

        The following parameters should only be changed if training the network

        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] tune the base/trunk network
            [True] keep base/trunk frozen
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = fix_vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = fix_alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]
            self.lins = nn.ModuleList(self.lins)

            # 假设权重文件中的层名是 lin3.model.1.weight，需映射到代码中的 lin3.model.0.weight
            weight_map1 = {
                'lin0.model.1.weight': 'lin0.model.0.weight',
                'lin1.model.1.weight': 'lin1.model.0.weight',
                'lin2.model.1.weight': 'lin2.model.0.weight',
                'lin3.model.1.weight': 'lin3.model.0.weight',
                'lin4.model.1.weight': 'lin4.model.0.weight',             
            }      


            if(pretrained):
                if(model_path is None):
                    import inspect
                    import os
                    model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth'%(version,net)))

                if(verbose):
                    print('Loading model from: %s'%model_path)
                
                state_dict = torch.load(model_path, map_location='cpu')  # 加载原始权重
                mapped_state_dict1 = {weight_map1[k]: v for k, v in state_dict.items()}
                self.load_state_dict(mapped_state_dict1, strict=False)  # strict=False允许部分匹配
                # self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)          

        if(eval_mode):
            self.eval()

# --- Helper functions from LPIPS ---
def normalize_tensor(in_feat, eps=1e-10):
    # Norms features per channel
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)

def spatial_average(in_tens, keepdim=True):
    # Averages features spatially
    return in_tens.mean([2, 3], keepdim=keepdim)

# --- End of LPIPS related code ---


def load_and_preprocess_image(image_path, device, target_size=None):
    """Loads an image, converts to RGB, resizes, and preprocesses for LPIPS."""
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # (width, height)

    transform = transforms.ToTensor()
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0).to(device)  # [1, C, H, W]

def main():
    parser = argparse.ArgumentParser(description="Calculate average LPIPS score between images in folder1 and all subdirectories of folder2.")
    parser.add_argument("--folder1", type=str, required=True, help="Path to the first folder of images (reference images).")
    parser.add_argument("--folder2", type=str, required=True, help="Path to the top directory containing subdirectories of images to compare against folder1.")
    parser.add_argument("--model", type=str, default="alex", choices=["alex", "vgg"], help="LPIPS model type: 'alex' or 'vgg'.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained LPIPS weights (.pth file) for linear layers. If None, uses LPIPS library defaults.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print(f"Initializing LPIPS model with backbone: {args.model}")

    try:
        lpips_model = FIX_LPIPS(net=args.model, pretrained=True, model_path=args.model_path, use_dropout=False, lpips=True, eval_mode=True, verbose=True)
    except Exception as e:
        print(f"An unexpected error occurred during model initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    lpips_model.to(args.device)
    lpips_model.eval()

    # Load reference images
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    folder1_images_paths = []
    for ext in image_extensions:
        folder1_images_paths.extend(glob.glob(os.path.join(args.folder1, ext)))

    if not folder1_images_paths:
        print(f"No images found in reference folder: {args.folder1}")
        return
    print(f"Found {len(folder1_images_paths)} reference images in {args.folder1}.")

    # Find subdirectories in folder2
    folder2_subdirs = [os.path.join(args.folder2, d) for d in os.listdir(args.folder2) if os.path.isdir(os.path.join(args.folder2, d))]

    if not folder2_subdirs:
        print(f"No subdirectories found in {args.folder2}. Please ensure it contains subdirectories with images.")
        return
    print(f"Found {len(folder2_subdirs)} subdirectories to process in {args.folder2}.")

    all_lpips_scores = []
    grand_total_processed_pairs = 0

    for sub_dir_path in folder2_subdirs:
        print(f"\n--- Processing subdirectory: {os.path.basename(sub_dir_path)} ---")
        current_subdir_processed_pairs = 0
        current_subdir_total_lpips = 0.0

        for img1_path in folder1_images_paths:
            img_filename = os.path.basename(img1_path)
            img2_path_in_subdir = os.path.join(sub_dir_path, img_filename)

            if not os.path.exists(img2_path_in_subdir):
                continue

            # Load original image to get size (H, W)
            try:
                with Image.open(img1_path) as img1_pil:
                    target_size = img1_pil.size[::-1]  # (height, width)
            except Exception as e:
                print(f"Error opening reference image {img1_path}: {e}")
                continue

            img1_tensor = load_and_preprocess_image(img1_path, args.device, target_size=target_size)
            img2_tensor = load_and_preprocess_image(img2_path_in_subdir, args.device, target_size=target_size)

            if img1_tensor is None or img2_tensor is None:
                print(f"Skipping pair due to loading error: {img_filename} in {os.path.basename(sub_dir_path)}")
                continue

            with torch.no_grad():
                try:
                    distance = lpips_model.forward(img1_tensor, img2_tensor, normalize=True)
                    dist_value = distance.item()
                except Exception as e:
                    print(f"Error calculating LPIPS for {img_filename} in {os.path.basename(sub_dir_path)}: {e}")
                    continue

            all_lpips_scores.append(dist_value)
            current_subdir_total_lpips += dist_value
            current_subdir_processed_pairs += 1
            grand_total_processed_pairs += 1

        if current_subdir_processed_pairs > 0:
            avg_lpips_subdir = current_subdir_total_lpips / current_subdir_processed_pairs
            print(f"Processed {current_subdir_processed_pairs} image pairs for subdirectory '{os.path.basename(sub_dir_path)}'. Average LPIPS: {avg_lpips_subdir:.4f}")
        else:
            print(f"No matching image pairs processed for subdirectory '{os.path.basename(sub_dir_path)}'.")

    if grand_total_processed_pairs > 0:
        overall_average_lpips = sum(all_lpips_scores) / grand_total_processed_pairs
        print(f"\n==================================================")
        print(f"Overall processed {grand_total_processed_pairs} image pairs across {len(folder2_subdirs)} subdirectories.")
        print(f"Overall Average LPIPS score: {overall_average_lpips:.4f}")
        print(f"==================================================")
    else:
        print("\nNo image pairs were processed across any subdirectories.")

if __name__ == "__main__":
    main()

# python -m lib.eval.lpips_eval --folder1 models/cele_plus_200 --folder2 temp/cele_plus_200/cradle_2  --model alex --model_path models/alex.pth 