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



class GRA(Attack):
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, model_learning_rate=0.0001,train_steps=[0,2,4,6,8]):
        super().__init__("GRA", model)
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
            x_p = F.softmax(output, dim=-1)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            current_gradient = x.grad.data
            x_i_p_list = []
            gradient_list = []
            for _ in range(self.max_iter):
                gauss = torch.rand_like(x) * 2 * (self.eps * self.beta) - self.eps * self.beta
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad = True)
                output_v3 = model(x_idct)
                loss = F.cross_entropy(output_v3, labels)
                loss.backward()
                gradient_list.append(x_idct.grad.data)
                x_i_p = F.softmax(output_v3, dim=-1)
                x_i_p_list.append(x_i_p)
            cossims = [(x_p * x_i_p).sum(1, keepdim=True) / (torch.sqrt((x_p ** 2).sum(1, keepdim=True)) * torch.sqrt((x_i_p ** 2).sum(1, keepdim=True))) for x_i_p in x_i_p_list]
            
            c_gradient = torch.zeros_like(current_gradient)
            for j in range(len(cossims)):
                cossim = cossims[j].unsqueeze(-1).unsqueeze(-1)
                gradient = gradient_list[j]
                c_gradient += (1-cossim) * current_gradient + cossim * gradient
            current_gradient = c_gradient / len(cossims)
            
            noise = self.momentum * grad + current_gradient / torch.abs(current_gradient).mean([1, 2, 3], keepdim=True)
            eqm = (torch.sign(grad) == torch.sign(noise)).float()
            grad = noise
            dim = torch.ones_like(images)  - eqm
            m = m * (eqm + dim * 0.94)
            x = x + self.alpha * torch.sign(grad) * m
            x = clip_by_tensor(x, images_min, images_max).detach()
            
            
        adv_img_np = x.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_func(images=adv_img_np,output_dir=output_dir[:-1]+f"_{i}/")
        return x.detach()
