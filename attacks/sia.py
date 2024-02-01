import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

import os

img_height, img_width = 224, 224
img_max, img_min = 1., 0

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

model_list = {'resnet18':models.resnet18, 
                'resnet101': models.resnet101,
                'resnext50': models.resnext50_32x4d,
                'densenet121': models.densenet121,
                'mobilenet': models.mobilenet_v2,
                'vit': models.vit_b_16,
                'swin': models.swin_t,
                'inceptionv3': models.inception_v3,
                }

def wrap_model(model):
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)

def load_images(input_dir, batch_size):
    images = [] 
    filenames = []
    idx = 0
    for filepath in os.listdir(input_dir):
        image = Image.open(os.path.join(input_dir,filepath))
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images.append(np.array(image).astype(np.float32)/255)
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            images = torch.from_numpy(np.array(images)).permute(0,3,1,2)
            yield filenames, images
            filenames = []
            images = []
            idx = 0
    if idx > 0:
        images = torch.from_numpy(np.array(images)).permute(0,3,1,2)
        yield filenames, images

def get_labels(names, f2l):
    labels = []
    for name in names:
        labels.append(f2l[name]-1)
    return torch.from_numpy(np.array(labels, dtype=np.int64))

def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)

def save_img(output_path,img):
    img = (img.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)

class Attack(object):
    """
    Base class for all attacks.
    """
    def __init__(self, attack, model, eps, targeted, random_start, norm, loss,device=None):
        """
        Initialize the hyperparameters
        Arguments:
            attack (str): the name of attack.
            model (torch.nn.Module): the surrogate model for attack.
            epsilon (float): the perturbation budget.
            targeted (bool): targeted/untargeted attack.
            random_start (bool): whether using random initialization for delta.
            norm (str): the norm of perturbation, l2/linfty.
            loss (str): the loss function.
            device (torch.device): the device for data. If it is None, the device would be same as model
        """
        if norm not in ['l2', 'linfty']:
            raise Exception("Unsupported norm {}".format(norm))
        self.attack = attack
        self.model = model
        self.epsilon = eps
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.device = next(model.parameters()).device if device is None else device
        self.loss = self.loss_function(loss)
        self.alpha = eps
        self.steps = 10
        self.decay = 1.0

    
    def forward(self, data, label, save_func,output_dir, **kwargs):
        """
        The general attack procedure
        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0.
        for _ in range(self.steps):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum,decay=self.decay)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        delta += data
        adv_img_np = delta.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_func(images=adv_img_np,output_dir=output_dir)
        return delta.detach()

    def get_logits(self, x, **kwargs):
        """
        The inference stage, which should be overridden when the attack need to change the models (e.g., ensemble-model attack, ghost, etc.) or the input (e.g. DIM, SIM, etc.)
        """
        return self.model(x)

    def get_loss(self, logits, label):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)
        

    def get_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad, momentum, decay=None, **kwargs):
        """
        The momentum calculation
        """
        return momentum * decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=10).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0,1).to(self.device)
                delta *= r/n*self.epsilon
            delta = clamp(delta, img_min-data, img_max-data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta


    def loss_function(self, loss):
        """
        Get the loss function
        """
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise Exception("Unsupported loss {}".format(loss))

    def transform(self, data, **kwargs):
        return data

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)
    
class MIFGSM(Attack):
    """
    MI-FGSM Attack
    'Boosting Adversarial Attacks with Momentum (CVPR 2018)'(https://arxiv.org/abs/1710.06081)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    """
    
    def __init__(self, model, eps=16/255, alpha=2/255, steps=10, decay=1., targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='MI-FGSM', **kwargs):
        super().__init__(attack, model, eps, targeted, random_start, norm, loss, device,**kwargs)
        self.alpha = alpha
        self.steps = steps
        self.decay = decay

class SIA(MIFGSM):
    """
    SIA Attack
    
    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of shuffled copies in each iteration.
        num_block (int): the number of block in the image.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, num_block=3
    """
    
    def __init__(self, model, eps=16/255, alpha=2/255, steps=10, decay=1., num_copies=20, num_block=3, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='SIM', **kwargs):
        super().__init__(model, eps, alpha, steps, decay, targeted, random_start, norm, loss, device, attack, **kwargs)
        self.num_copies = num_copies
        self.num_block = num_block
        self.kernel = self.gkern()
        self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180, self.scale, self.add_noise,self.dct,self.drop_out]
        
    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2,3))
    
    def scale(self, x):
        return torch.rand(1)[0] * x
    
    def resize(self, x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor)+1
        new_w = int(w * scale_factor)+1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x
    
    def dct(self, x):
        """
        Discrete Fourier Transform
        """
        dctx = dct.dct_2d(x) #torch.fft.fft2(x, dim=(-2, -1))
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:,:] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx # * self.mask.reshape(1, 1, w, h)
        idctx = dct.idct_2d(dctx)
        return idctx
    
    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def drop_out(self, x):
        
        return F.dropout2d(x, p=0.1, training=True)

    def blocktransform(self, x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0,] + np.random.choice(list(range(1, h)), self.num_block-1, replace=False).tolist() + [h,]
        x_axis = [0,] + np.random.choice(list(range(1, w)), self.num_block-1, replace=False).tolist() + [w,]
        y_axis.sort()
        x_axis.sort()
        
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy

    def transform(self, x, **kwargs):
        """
        Scale the input for BlockShuffle
        """
        return torch.cat([self.blocktransform(x) for _ in range(self.num_copies)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return self.loss(logits, label.repeat(self.num_copies))