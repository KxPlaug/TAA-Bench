import torch
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
from attacks.attack import Attack
import numpy as np
import copy
features = None

class FGSM:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=10, alpha=0.001):
        alpha = self.epsilon / num_steps
        dt = data.clone().detach().requires_grad_(True)
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            data_grad_sign = dt.grad.data.sign()
            adv_data = dt + alpha * data_grad_sign
            total_grad = adv_data - data
            total_grad = torch.clamp(
                total_grad, -self.epsilon, self.epsilon)
            dt.data = torch.clamp(
                data + total_grad, self.data_min, self.data_max)
        return dt

def hook_feature(module, input, output):
    global features
    features = output


def image_transform(x):
    return x


def get_NAA_loss(adv_feature, base_feature, weights):
    gamma = 1.0
    attribution = (adv_feature - base_feature) * weights
    blank = torch.zeros_like(attribution)
    positive = torch.where(attribution >= 0, attribution, blank)
    negative = torch.where(attribution < 0, attribution, blank)
    # Transformation: Linear transformation performs the best
    balance_attribution = positive + gamma * negative
    loss = torch.sum(balance_attribution) / \
        (base_feature.shape[0]*base_feature.shape[1])
    return loss


def normalize(grad, opt=2):
    if opt == 0:
        nor_grad = grad
    elif opt == 1:
        abs_sum = torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        nor_grad = grad/abs_sum
    elif opt == 2:
        square = torch.sum(torch.square(grad), dim=(1, 2, 3), keepdim=True)
        nor_grad = grad/torch.sqrt(square)
    return nor_grad


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    # x = F.pad(x, (kern_size, kern_size, kern_size, kern_size, 0, 0), "constant")
    x = F.conv2d(x, stack_kern, padding = (kern_size, kern_size), groups=3)
    return x


"""Input diversity: https://arxiv.org/abs/1803.06978"""


def input_diversity(x, resize_rate=1.15, diversity_prob=0.7):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize,
                        size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(),
                            size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(),
                             size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(
    ), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    # resize back to [img_size, img_size]
    ret = F.interpolate(
        ret, size=[img_size, img_size], mode='bilinear', align_corners=False)
    return ret


P_kern, kern_size = project_kern(3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DANAA(Attack):
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10,ieps=16/255,method="DIPIBIM"):
        super().__init__("DANAA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.ieps = ieps
        self.supported_mode = ['default', 'targeted']
        self.momentum = 1.0
        if model[1].__class__.__name__ == "Inception3":
            self.target_layer = "1.Mixed_5b"
            for name, module in self.model.named_modules():
                if name == "1.Mixed_5b":
                    module.register_forward_hook(hook_feature)
                    break
        elif model[1].__class__.__name__ == "InceptionV4":
            self.target_layer = "1.features.5"
            for name, module in self.model.named_modules():
                if name == "1.features.5":
                    module.register_forward_hook(hook_feature)
                    break
        elif model[1].__class__.__name__ == "ResNet":
            self.target_layer = "1.layer3.8"
            for name, module in self.model.named_modules():
                if name == "1.layer3.8":
                    module.register_forward_hook(hook_feature)
                    break
        elif model[1].__class__.__name__ == "InceptionResNetV2":
            self.target_layer = "1.conv2d_4a.relu"
            for name, module in self.model.named_modules():
                if name == "1.conv2d_4a.relu":
                    module.register_forward_hook(hook_feature)
                    break
        self.ens = 30.0
        self.gamma = 0.5
        self.method = method
        self.kernel_name = "gaussian"
        self.len_kernel = 15
        self.nsig = 3
        self.amplification_factor = 2.5
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        
    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    
    
    def bim_method(self, model,images,y,labels,weight_np):
        images_base = image_transform(images.clone())
        for l in range(int(self.ens)):
            c = images_base + (torch.randn_like(images)*0.2 + 0)
            c = c.requires_grad_(True)
            if self.method.startswith("DIPI"):
                logits = model(input_diversity(c))
            else:
                logits = model(c)
            loss = F.cross_entropy(logits, y)
            x_grad_real = torch.autograd.grad(
                loss, c)[0]
            images_base = images_base + 0.0025 * x_grad_real.sign()
            if self.method.startswith("DIPI"):
                logits = model(input_diversity(c))
            else:
                logits = model(c)
            if weight_np is None:
                weight_np = torch.autograd.grad(
                    F.softmax(logits, dim=-1)*labels, features, grad_outputs=torch.ones_like(logits*labels))[0]
            else:
                weight_np += torch.autograd.grad(
                    F.softmax(logits, dim=-1)*labels, features, grad_outputs=torch.ones_like(logits*labels))[0]
        return weight_np
    

    def forward(self, x, y, save_func,output_dir):
        images = x.clone().detach().to(device)
        model = copy.deepcopy(self.model)
        y = y.to(device)
        labels = torch.nn.functional.one_hot(y, 1000).to(device)
        adv_images = x.clone().detach().to(device)
        grad_np = torch.zeros_like(images)
        amplification_np = torch.zeros_like(images)
        weight_np = None
        for iter in range(self.steps):
            adv_images.requires_grad = True
            if iter == 0:
                if self.ens == 0:
                    adv_images = image_transform(adv_images)
                    if "DI" in self.method:
                        logits = model(input_diversity(adv_images))
                    else:
                        logits = model(adv_images)
                    if weight_np is None:
                        weight_np = torch.autograd.grad(
                            F.softmax(logits)*labels, features, grad_outputs=torch.ones_like(F.softmax(logits)*labels))[0]
                    else:
                        weight_np += torch.autograd.grad(
                            F.softmax(logits)*labels, features, grad_outputs=torch.ones_like(F.softmax(logits)*labels))[0]
                        
                weight_np = self.bim_method(model,images,y,labels,weight_np)
                weight_np = -normalize(weight_np, 2)
            images_base = image_transform(torch.zeros_like(images))
            if self.method.startswith("DIPI"):
                _ = model(input_diversity(images_base))
            else:
                _ = model(images_base)
            base_feamap = features
            if self.method.startswith("DIPI"):
                _ = model(input_diversity(adv_images))
            else:
                _ = model(adv_images)
            adv_feamap = features
            loss = get_NAA_loss(adv_feamap, base_feamap, weight_np)
            grad = torch.autograd.grad(
                loss, adv_images)[0]

            grad = grad / torch.mean(torch.abs(grad), [1, 2, 3], keepdim=True)
            grad = self.momentum * grad_np + grad
            grad_np = grad.clone().detach().to(device)

            if self.method.startswith("DIPI"):
                # amplification factor
                alpha_beta = self.alpha * self.amplification_factor
                gamma = self.gamma * alpha_beta
                # Project cut noise
                amplification_np += alpha_beta * torch.sign(grad)
                cut_noise = torch.clip(abs(amplification_np) -
                                       self.eps, 0.0, 10000.0) * torch.sign(amplification_np)
                projection = gamma * \
                    torch.sign(project_noise(cut_noise, P_kern, kern_size))
                amplification_np += projection
                adv_images = adv_images + alpha_beta * \
                    torch.sign(grad) + projection
            else:
                adv_images = adv_images + self.alpha * torch.sign(grad)

            adv_images = torch.clip(
                adv_images, images - self.eps, images + self.eps)
            adv_images = torch.clip(adv_images, 0.0, 1.0)
            adv_images = adv_images.detach()
            
        adv_img_np = adv_images.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_func(images=adv_img_np,output_dir=output_dir[:-1]+f"_{iter}/")
        return adv_images

 