import PIL.Image

from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator
from carvekit.ml.arch.tracerb7.tracer import TracerDecoder
from carvekit.utils.models_utils import get_precision_autocast, cast_network
from carvekit.utils.image_utils import load_image, convert_image
from carvekit.utils.pool_utils import thread_pool_processing, batch_generator
import torch
from pathlib import Path
from typing import List, Union
from carvekit.utils.mask_utils import apply_mask
import torch.nn.functional as F
import os
from tqdm import tqdm

def my_data_preprocessing(self, input_tensor: torch.Tensor,):
    resized_tensor = F.interpolate(
        input=input_tensor.unsqueeze(0),  # unsqueeze 添加 batch 维度
        size=self.input_image_size,
        mode='bilinear',
        align_corners=False
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Convert mean and std to tensors
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    normalized_tensor = resized_tensor.sub_(mean_tensor).div_(std_tensor)

    return normalized_tensor


def seg_call(
    self, images: List[Union[torch.Tensor, PIL.Image.Image, str, Path]],
) -> List[Union[torch.Tensor, PIL.Image.Image]]:
    """
    Passes input images though neural network and returns segmentation masks as PIL.Image.Image instances

    Args:
        images: input images

    Returns:
        segmentation masks as for input images, as PIL.Image.Image instances

    """
    collect_masks = []
    autocast, dtype = get_precision_autocast(device=self.device, fp16=self.fp16)
    with autocast:
        cast_network(self, dtype)
        for image_batch in batch_generator(images, self.batch_size):
            if isinstance(image_batch[0], torch.Tensor):
                batches = torch.vstack(
                    thread_pool_processing(self.my_data_preprocessing, images,)
                )
            else:
                images = thread_pool_processing(
                    lambda x: convert_image(load_image(x)), image_batch
                )
                batches = torch.vstack(
                    thread_pool_processing(self.data_preprocessing, images)
                )
            with torch.no_grad():
                batches = batches.to(self.device)
                masks = super(TracerDecoder, self).__call__(batches)

            if self.return_pil:    
                masks_cpu = masks.cpu()
                del batches, masks
                masks = thread_pool_processing(
                    lambda x: self.data_postprocessing(masks_cpu[x], images[x]),
                    range(len(images)),
                )
            collect_masks += masks

    return collect_masks


def interface_call(
        self, images: List[Union[torch.Tensor, PIL.Image.Image, str, Path]]
    ) -> List[Union[torch.Tensor, PIL.Image.Image]]:
        """
        Removes the background from the specified images.

        Args:
            images: list of input images

        Returns:
            List of images without background as PIL.Image.Image instances
        """
        if isinstance(images[0], torch.Tensor) is False:
            images = thread_pool_processing(load_image, images)

        if self.preprocessing_pipeline is not None:
            masks: List[Image.Image] = self.preprocessing_pipeline(
                interface=self, images=images
            )
        else:
            masks: List[Image.Image] = self.segmentation_pipeline(images=images)

        if self.postprocessing_pipeline is not None:
            images: List[Image.Image] = self.postprocessing_pipeline(
                images=images, masks=masks
            )
        else:
            images = list(
                map(
                    lambda x: apply_mask(
                        image=images[x], mask=masks[x], device=self.device
                    ),
                    range(len(images)),
                )
            )
        return images


class CarveKit():
    def __init__(self, seg_net_path, fba_path, device, weight_dtpye):
        # Check doc strings for more information
        self.seg_net = TracerUniversalB7(device=device,
                    batch_size=1, model_path=seg_net_path)
        
        setattr(TracerUniversalB7, '__call__', seg_call)
        setattr(TracerUniversalB7, 'my_data_preprocessing', my_data_preprocessing)


        self.fba = FBAMatting(device=device,
                        input_tensor_size=2048,
                        batch_size=1, load_pretrained=False)

        self.fba.load_state_dict(torch.load(fba_path, map_location=device))

        self.seg_net = self.seg_net.to(device, dtype=weight_dtpye)
        self.fba = self.fba.to(device, dtype=weight_dtpye)


        self.trimap = TrimapGenerator()

        self.preprocessing = PreprocessingStub()

        self.postprocessing = MattingMethod(matting_module=self.fba,
                                    trimap_generator=self.trimap,
                                    device=device)

        self.interface = Interface(pre_pipe=self.preprocessing,
                            post_pipe=self.postprocessing,
                            seg_pipe=self.seg_net)
        setattr(Interface, '__call__', interface_call)

        self.seg_net.return_pil = True


        
    def __call__(self, image: List,):
        result = self.interface(image)
        return result

import torchvision.transforms as transforms
from PIL import Image

def process_dataset(carveKit, dataset_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in tqdm(files):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                new_img_path = os.path.join(output_dir, file)
                if os.path.exists(new_img_path):
                    continue

                with Image.open(img_path).convert("RGB") as img:
                    result = carveKit([img])[0]                       

                    result.convert("RGB").save(new_img_path)



if __name__ == "__main__":
    carvekit = CarveKit("/home/u2120240694/data/model/tracer_b7.pth", "/home/u2120240694/data/model/fba_matting.pth", "cuda:0", torch.float32)
    process_dataset(carvekit, "others/celeba_hq_256", "others/celeba_hq_256_wo_bg")
    
    

                   