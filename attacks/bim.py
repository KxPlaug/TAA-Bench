import torch
import torch.nn as nn
import copy
from attacks.attack import Attack
import numpy as np
import torch.nn.functional as F

class BIM(Attack):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels, save_func, output_dir):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        model = copy.deepcopy(self.model)

        loss = nn.CrossEntropyLoss()
        
        ori_images = images.clone().detach()

        for i in range(self.steps):
            images.requires_grad = True
            outputs = model(images)

            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_images = images + self.alpha*grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images \
                + (adv_images < a).float()*a
            c = (b > ori_images+self.eps).float()*(ori_images+self.eps) \
                + (b <= ori_images + self.eps).float()*b
            images = torch.clamp(c, max=1).detach()
            
        adv_img_np = images.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_func(images=adv_img_np,output_dir=output_dir[:-1]+f"_{i}/")
                
        return images
