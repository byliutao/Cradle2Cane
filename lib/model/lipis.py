from lpips import LPIPS, ScalingLayer, NetLinLayer, pn, upsample, spatial_average
from lpips.pretrained_networks import alexnet, vgg16
from torchvision import models as tv
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.vgg import VGG16_Weights
import torch
import torch.nn as nn
import os
import lpips
import torch
import argparse
import numpy as np
import PIL

from lib.utils.common_utils import lpips_loss

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p0','--path0', type=str, default="others/48_male_27_White.jpg")
    parser.add_argument('-p1','--path1', type=str, default="temp/infer4/48_male_27_White/48_male_27_White.jpg_27.png")
    # parser.add_argument('-v','--version', type=str, default='0.1')

    opt = parser.parse_args()

    # Initializing the model
    loss_fn_alex = FIX_LPIPS(net='alex', pretrained=True, model_path="/data/model/weights/v0.1/alex.pth", use_dropout=False, lpips=True) # best forward scores
    loss_fn_vgg = FIX_LPIPS(net='vgg', pretrained=True, model_path="/data/model/weights/v0.1/vgg.pth", use_dropout=False, lpips=True) # closer to "traditional" perceptual loss, when used for optimization
    # loss_fn_alex = loss_fn_vgg

    loss_fn_alex.cuda()

    # Load images
    img0 = lpips.im2tensor(lpips.load_image(opt.path0)).to("cuda") # RGB image from [-1,1]
    img1 = lpips.im2tensor(lpips.load_image(opt.path1)).to("cuda")

    
    import torchvision
    transform = torchvision.transforms.ToTensor()
    img0_pil = PIL.Image.open(opt.path0)
    img1_pil = PIL.Image.open(opt.path1)
    img0_pil = transform(img0_pil).to("cuda")
    img1_pil = transform(img1_pil).to("cuda")

    # Compute distance
    dist01 = loss_fn_alex.forward(img0, img1, normalize=False)
    print('Distance: %.4f'%dist01)

    # Compute distance
    dist01_pil = loss_fn_alex.forward(img0_pil, img1_pil, normalize=True)
    print('Distance: %.4f'%dist01_pil)

    loss = lpips_loss(loss_fn_alex, [img0_pil], [img1_pil])
    print(loss)

    print(torch.mean(img0_pil))
    print(torch.mean(img1_pil))
    



