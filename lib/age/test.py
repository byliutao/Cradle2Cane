import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

from lib.age.model import get_model
from lib.age.defaults import _C as cfg


def get_args():
    parser = argparse.ArgumentParser(description="Age prediction on a single image",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_path", type=str, default="/data/dataset/AgeDB/148_PaulmcCartney_17_m.jpg", help="Path to the input image")
    parser.add_argument("--resume", type=str, default="/data/model/age_predict.pth", help="Model weight to be used")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def predict_age(img_tensor, model, device, reshape=False):
    """
    Predicts the age for a single image
    
    Args:
        img_tensor: Single image tensor with shape [1, C, H, W]
        model: Trained model
        device: Device to run inference on
    
    Returns:
        predicted_age: The predicted age for the image
    """
    model.eval()
    
    if reshape:
        transform = transforms.Compose([
            transforms.Resize((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE)),
        ])    
        img_tensor = transform(img_tensor)  # Add batch dimension


    with torch.no_grad():
        # Ensure input is on the correct device
        img_tensor = img_tensor.to(device)
        
        # Forward pass
        outputs = model(img_tensor)
        
        # Convert to probabilities
        probs = F.softmax(outputs, dim=-1)
        
        # Calculate expected age
        ages = torch.arange(0, 101).float().to(device)
        predicted_age = (probs * ages).sum(dim=-1)
    
    return predicted_age


def load_model(path, device, weight_dtype,):
    cfg.freeze()
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    model = model.to(device, dtype=weight_dtype)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    cudnn.benchmark = True

    return model


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    model = load_model(args.resume, "cuda", torch.float32)
    transform = transforms.Compose([
        transforms.Resize((cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE)),
        transforms.ToTensor(),
    ])    
    img = Image.open(args.image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Load image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    age = predict_age(img_tensor, model, "cuda")
    print(int(age))



if __name__ == '__main__':
    main()