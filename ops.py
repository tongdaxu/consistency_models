'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml

import torch.nn as nn
from torch.nn import functional as F

import torchvision

import numpy as np
import torch
from math import pi

import scipy

from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import os 
from torchvision import transforms
from roomlayout import LayoutSeg
from roomsegmentation import SegmentationModule, ModelBuilder

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from roomtext import blip_itm, blip_decoder
from gg18 import ScaleHyperpriorSTE
from torchvision.models import resnet50, ResNet50_Weights

class Resizer(nn.Module):
    def __init__(self, in_shape, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
        super(Resizer, self).__init__()
        scale_factor, output_shape = self.fix_scale_and_size(in_shape, output_shape, scale_factor)
        method, kernel_width = {
            "cubic": (cubic, 4.0),
            "lanczos2": (lanczos2, 4.0),
            "lanczos3": (lanczos3, 6.0),
            "box": (box, 1.0),
            "linear": (linear, 2.0),
            None: (cubic, 4.0)  # set default interpolation method as cubic
        }.get(kernel)

        antialiasing *= (np.any(np.array(scale_factor) < 1))

        sorted_dims = np.argsort(np.array(scale_factor))
        self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

        field_of_view_list = []
        weights_list = []
        for dim in self.sorted_dims:
            weights, field_of_view = self.contributions(in_shape[dim], output_shape[dim], scale_factor[dim], method,
                                                        kernel_width, antialiasing)

            weights = torch.tensor(weights.T, dtype=torch.float32)

            weights_list.append(
                nn.Parameter(torch.reshape(weights, list(weights.shape) + (len(scale_factor) - 1) * [1]),
                             requires_grad=False))
            field_of_view_list.append(
                nn.Parameter(torch.tensor(field_of_view.T.astype(np.int32), dtype=torch.long), requires_grad=False))

        self.field_of_view = nn.ParameterList(field_of_view_list)
        self.weights = nn.ParameterList(weights_list)

    def forward(self, in_tensor):
        x = in_tensor
        for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):
            x = torch.transpose(x, dim, 0)
            x = torch.sum(x[fov] * w, dim=0)
            x = torch.transpose(x, dim, 0)
        return x

    def fix_scale_and_size(self, input_shape, output_shape, scale_factor):
        if scale_factor is not None:
            if np.isscalar(scale_factor) and len(input_shape) > 1:
                scale_factor = [scale_factor, scale_factor]
            scale_factor = list(scale_factor)
            scale_factor = [1] * (len(input_shape) - len(scale_factor)) + scale_factor
        if output_shape is not None:
            output_shape = list(input_shape[len(output_shape):]) + list(np.uint(np.array(output_shape)))
        if scale_factor is None:
            scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)
        if output_shape is None:
            output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))
        return scale_factor, output_shape

    def contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
        kernel_width *= 1.0 / scale if antialiasing else 1.0
        out_coordinates = np.arange(1, out_length + 1)
        shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2
        match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)
        left_boundary = np.floor(match_coordinates - kernel_width / 2)
        expanded_kernel_width = np.ceil(kernel_width) + 2
        field_of_view = np.squeeze(
            np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))
        weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)
        sum_weights = np.sum(weights, axis=1)
        sum_weights[sum_weights == 0] = 1.0
        weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)
        mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]
        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        weights = np.squeeze(weights[:, non_zero_out_pixels])
        field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])
        return weights, field_of_view


def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        for name, param in self.conv.named_parameters():
            param.requires_grad = False

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='roomlayout')
class Layout(LinearOperator):
    def __init__(self, weight_path, device):
        self.model = LayoutSeg.load_from_checkpoint(weight_path, backbone='resnet101').to(device)
        self.model.freeze()

    def forward(self, data, **kwargs):
        scores, _ = self.model(data)
        assert 'mode' in kwargs
        if kwargs['mode'] == 'init':
            return torch.argmax(scores, dim=1, keepdim=True)
        else:
            return scores

    def transpose(self, data):
        return data
    
@register_operator(name='roomsegmentation')
class Segmentation(LinearOperator):
    def __init__(self,device):
        self.encoder = ModelBuilder.build_encoder(arch="mobilenetv2dilated",fc_dim=320,weights="/NEW_EDS/JJ_Group/zhuzr/xutd_cm/encoder_epoch_20.pth").to('cuda')
        self.decoder = ModelBuilder.build_decoder(arch="c1_deepsup",fc_dim=320,num_class=150,weights="/NEW_EDS/JJ_Group/zhuzr/xutd_cm/decoder_epoch_20.pth",use_softmax=True).to('cuda')
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.decoder.named_parameters():
            param.requires_grad = False
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    def forward(self, data, **kwargs):
        data = (data + 1) / 2.0
        data = self.transform(data)
        pred = self.decoder(self.encoder(data, return_feature_maps=True), segSize=(256,256))
        assert 'mode' in kwargs
        if kwargs['mode'] == 'init':
            return torch.argmax(pred, dim=1, keepdim=True)
        else:
            return pred
    
    def transpose(self, data):
        return data

@register_operator(name='roomtext')
class Image2Text(LinearOperator):
    def __init__(self,device) -> None:
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform_test = transforms.Compose([transforms.Resize((384, 384),interpolation=InterpolationMode.BICUBIC),self.normalize,])
        self.itm_model = blip_itm(pretrained='/NEW_EDS/JJ_Group/zhuzr/huggingface/BLIP/model_base_retrieval_coco.pth', image_size=384, vit='base')
        self.itm_model.eval()
        self.itm_model = self.itm_model.to(device='cuda')
        self.blip_decoder_model = blip_decoder(pretrained='/NEW_EDS/JJ_Group/zhuzr/huggingface/BLIP/model_base_caption_capfilt_large.pth', image_size=384, vit='base')
        self.blip_decoder_model.eval()
        self.blip_decoder_model = self.blip_decoder_model.to(device='cuda')
    def forward(self, data, **kwargs):
        data = (data + 1.0)/2
        data = self.transform_test(data)
        if kwargs['mode'] == 'init':
            return self.blip_decoder_model.generate(data, sample=False, num_beams=3, max_length=20, min_length=5)
        else:
            itm_output = self.itm_model(data,kwargs['caption'],match_head='itm')
            itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
            return - itm_score
        
    def transpose(self,data):
        return None

@register_operator(name='catcls')
class CatClassification(LinearOperator):
    def __init__(self,device):
        self.net = torch.load("/NEW_EDS/JJ_Group/zhuzr/icml24/dataset/animal-breed-classification/cat_breed_vgg16.pth") 
       
    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0
        score = self.net(data)    
        score = F.softmax(score, dim=1)
        if kwargs['mode'] == 'init':
            result = torch.zeros_like(score)
            index = torch.argmax(score, dim=1, keepdim=True)
            return index
        else:
            return score
        
    def transpose(self,data):
        return None


@register_operator(name='catcls2')
class CatClassification2(LinearOperator):
    def __init__(self, device):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
        self.model.eval()
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform_test = transforms.Compose([transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),self.normalize,])
    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0
        data = self.transform_test(data)
        uscore = self.model(data)
        score = F.softmax(uscore, dim=1)
        if kwargs['mode'] == 'init':
            index = torch.argmax(score, dim=1, keepdim=True)
            return index
        else:
            return score
        
    def transpose(self,data):
        return None

@register_operator(name='ddrmsr')
class DDRMSRConv(LinearOperator):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                        dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                        dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel = None, channels = 3, img_dim =256, device = 'cuda', stride = 4):
        
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
            else:
                return 0
            
        factor = 4
        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1/factor)*(i - np.floor(factor*4/2) +0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to('cuda')
        
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        #build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride//2, img_dim + stride//2, stride):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                j_effective = j
                #reflective padding
                if j_effective < 0: j_effective = -j_effective-1
                if j_effective >= img_dim: j_effective = (img_dim - 1) - (j_effective - img_dim)
                #matrix building
                H_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1), self.singulars_small.reshape(1, small_dim)).reshape(small_dim**2)
        #permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)] + \
                                  [self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.small_dim**2, self.channels, device=vec.device)
        temp[:, :self.small_dim**2, :] = vec.clone().reshape(vec.shape[0], self.small_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    
    def H(self, vec):
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))

    def forward(self, data, **kwargs):
        y = self.H(data)
        return y

@register_operator(name='ddrmblur')
class DDRMDeblurring(LinearOperator):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO = 3e-2):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(H_small, some=False)
        #ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
    
    def H(self, vec):
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def Ht(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def H_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))
    
    def forward(self, data, **kwargs):
        y = self.H(data)
        return y


model_paths = [
    '/home/xutd/.cache/torch/hub/checkpoints/bmshj2018-hyperprior-1-7eb97409.pth.tar',
    '/home/xutd/.cache/torch/hub/checkpoints/bmshj2018-hyperprior-2-93677231.pth.tar',
    '/home/xutd/.cache/torch/hub/checkpoints/bmshj2018-hyperprior-3-6d87be32.pth.tar',
    '/home/xutd/.cache/torch/hub/checkpoints/bmshj2018-hyperprior-4-de1b779c.pth.tar',
    '/home/xutd/.cache/torch/hub/checkpoints/bmshj2018-hyperprior-5-f8b614e1.pth.tar',
    '/home/xutd/.cache/torch/hub/checkpoints/bmshj2018-hyperprior-6-1ab9c41e.pth.tar',
    '/home/xutd/.cache/torch/hub/checkpoints/bmshj2018-hyperprior-7-3804dcbd.pth.tar',
    '/home/xutd/.cache/torch/hub/checkpoints/bmshj2018-hyperprior-8-a583f0cf.pth.tar',
]
Ns, Ms = [128,128,128,128,128,192,192,192], [192,192,192,192,192,320,320,320]

@register_operator(name='gg18')
class CodecOperator(LinearOperator):
    def __init__(self, q, device):
        self.codec = ScaleHyperpriorSTE(Ns[q-1], Ms[q-1])
        self.codec.load_state_dict_gg18(torch.load(model_paths[q - 1]))
        self.codec = self.codec.cuda()
        self.codec.eval()
        print("load gg18 q: {}".format(q))
        
    def forward(self, data, **kwargs):
        out = self.codec((data + 1.0) / 2.0)
        return (out["x_bar"] * 2.0) - 1.0

    def transpose(self,data):
        return None

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return min(len(self.fpaths), 1000)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

@register_dataset(name='lsunlayout')
class LSUNLayout(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return min(len(self.fpaths), 1000)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath)
        
        if self.transforms is not None:
            img = self.transforms(img)
        img = torch.round(img * 5) - 1
        return img # [0,1,2,3,4]

if __name__ == "__main__":
    trans = transforms.Compose([transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                torchvision.transforms.CenterCrop(256),
                                transforms.ToTensor()])
    dataset = LSUNLayout(root='/NEW_EDS/JJ_Group/xutd/lsun-room/data/lsun_room/labelpng',
                         transforms=trans)
    dataloader = DataLoader(dataset, 
                            1)

    for y in dataloader:
        print(y.dtype)
        assert(0)
