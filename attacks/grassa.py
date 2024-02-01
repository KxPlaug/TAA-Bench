import torch
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
from attacks.attack import Attack
import numpy as np
import copy
from torch.autograd import Variable as V
from attacks.dct import *

def clip_by_tensor(t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

"""Input diversity: https://arxiv.org/abs/1803.06978"""
def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret
T_kernel = gkern(7, 3)


class GRASSA(Attack):
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, model_learning_rate=0.0001,train_steps=[0,2,4,6,8]):
        super().__init__("GRASSA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.momentum = 1.0
        self.beta = 3.5
        self.rho = 0.5
        self.max_iter = 20
        self.model_learning_rate = model_learning_rate
        self.train_steps = train_steps


    def forward(self, images, labels, save_func,output_dir):
        m = torch.ones_like(images) * 10 / 9.4
        
        x = images.clone()
        images_min = clip_by_tensor(images - self.eps, 0.0, 1.0)
        images_max = clip_by_tensor(images + self.eps, 0.0, 1.0)
        model = copy.deepcopy(self.model)
        grad = torch.zeros_like(images)
        
        for i in range(self.steps):
                
            x = V(x.detach(), requires_grad=True)
            output = model(x)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            current_gradient = x.grad.data
            avg_gradient = 0
            for _ in range(self.max_iter):
                gauss = torch.rand_like(x) * 2 * (self.eps * self.beta) - self.eps * self.beta
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad = True)
                output_v3 = model(DI(x_idct))
                loss = F.cross_entropy(output_v3, labels)
                loss.backward()
                avg_gradient += x_idct.grad.data
            avg_gradient = avg_gradient / self.max_iter
            cossim = (current_gradient * avg_gradient).sum([1, 2, 3], keepdim=True) / (
                        torch.sqrt((current_gradient ** 2).sum([1, 2, 3], keepdim=True)) * torch.sqrt(
                    (avg_gradient ** 2).sum([1, 2, 3], keepdim=True)))
            current_gradient = cossim * current_gradient + (1 - cossim) * avg_gradient
            current_gradient = F.conv2d(current_gradient, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
            current_gradient = current_gradient / torch.abs(current_gradient).mean([1, 2, 3], keepdim=True)
            current_gradient = self.momentum * grad + current_gradient
            eqm = (torch.sign(grad) == torch.sign(current_gradient)).float()
            grad = current_gradient
            dim = torch.ones_like(images)  - eqm
            m = m * (eqm + dim * 0.94)
            x = x + self.alpha * torch.sign(grad) * m
            x = clip_by_tensor(x, images_min, images_max)
            
            
        adv_img_np = x.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_func(images=adv_img_np,output_dir=output_dir[:-1]+f"_{i}/")
        return x.detach()
