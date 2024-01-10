import torch
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
from attacks.attack import Attack
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from captum.attr import IntegratedGradients
import torch
import torch.nn as nn

def compute_integrated_gradient(batch_x, batch_blank, model, idx):
    mean_grad = 0
    n = 50

    for i in range(1, n + 1):
        x = batch_blank + i / n * (batch_x - batch_blank)
        x.requires_grad = True
        y = model(x)
        y = torch.diag(y[:, idx])
        (grad,) = torch.autograd.grad(y, x,grad_outputs=torch.ones_like(y))
        mean_grad += grad / n

    integrated_gradients = (batch_x - batch_blank) * mean_grad

    return integrated_gradients

class IGAttack(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10,decay=1.0):
        super().__init__("IGAttack", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.model = model
        self.mu = decay

    def forward(self, images, labels, save_func,output_dir):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        last_g = torch.zeros_like(images).detach().to(self.device)
        for iter in range(self.steps):
            # delta_t = -self.ig.attribute(adv_images, baselines=torch.zeros_like(adv_images), target=labels).float()
            delta_t = -compute_integrated_gradient(adv_images, torch.zeros_like(adv_images), self.model, labels)
            g = self.mu * last_g + delta_t / (delta_t).abs().mean(dim=[1,2,3],keepdim=True)
            last_g = g
            adv_images = adv_images.detach() + self.alpha * g.sign()
            adv_images = torch.max(torch.min(adv_images, images + self.eps), images - self.eps).detach()
            adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

        adv_img_np = adv_images.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_func(images=adv_img_np,output_dir=output_dir[:-1]+f"_{iter}/")
        return adv_images
    