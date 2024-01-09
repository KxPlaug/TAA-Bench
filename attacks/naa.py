import torch
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
from attacks.attack import Attack
import numpy as np
import copy

features = None

    
    
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
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / \
        (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    x = torch.nn.functional.pad(
        x, (kern_size, kern_size, kern_size, kern_size), "constant", 0)
    x = torch.nn.functional.conv2d(
        x, stack_kern, stride=1, padding=0, groups=3)
    return x


"""Input diversity: https://arxiv.org/abs/1803.06978"""


def input_diversity(x, resize_rate=1.15, diversity_prob=0.5):
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
    return ret


P_kern, kern_size = project_kern(3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NAA(Attack):
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, model_learning_rate=0.0001,train_steps=[0,2,4,6,8]):
        super().__init__("NAA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.momentum = 1.0
        self.attack_method = "NAA"
        self.ens = 30.0
        self.gamma = 0.5
        # for name, module in model.named_modules():
        #     if name == "1.Mixed_5b":
        #         module.register_forward_hook(hook_feature)
        #         break
        if model[1].__class__.__name__ == "Inception3":
            for name, module in self.model.named_modules():
                if name == "1.Mixed_5b":
                    module.register_forward_hook(hook_feature)
                    break
        elif model[1].__class__.__name__ == "InceptionV4":
            for name, module in self.model.named_modules():
                if name == "1.features.5":
                    module.register_forward_hook(hook_feature)
                    break
        elif model[1].__class__.__name__ == "ResNet":
            for name, module in self.model.named_modules():
                if name == "1.layer3.8":
                    module.register_forward_hook(hook_feature)
                    break
        elif model[1].__class__.__name__ == "InceptionResNetV2":
            for name, module in self.model.named_modules():
                if name == "1.conv2d_4a.relu":
                    module.register_forward_hook(hook_feature)
                    break
        self.train_steps = train_steps
        self.model_learning_rate = model_learning_rate

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
                    if "DI" in self.attack_method:
                        logits = model(input_diversity(adv_images))
                    else:
                        logits = model(adv_images)
                    if weight_np is None:
                        weight_np = torch.autograd.grad(
                            F.softmax(logits)*labels, features, grad_outputs=torch.ones_like(F.softmax(logits)*labels))[0]
                    else:
                        weight_np += torch.autograd.grad(
                            F.softmax(logits)*labels, features, grad_outputs=torch.ones_like(F.softmax(logits)*labels))[0]

                for l in range(int(self.ens)):
                    x_base = np.array([0.0, 0.0, 0.0])
                    x_base = image_transform(x_base)
                    images_base = image_transform(images.clone())
                    images_base += (torch.randn_like(images)*0.2 + 0)
                    images_base = images_base.cpu().numpy().transpose(0, 2, 3, 1)
                    images_base = images_base * (1 - l / self.ens) + \
                        (l / self.ens) * x_base
                    images_base = torch.from_numpy(
                        images_base.transpose(0, 3, 1, 2)).float().to(device)
                    # print(images_base)
                    if "DI" in self.attack_method:
                        logits = model(input_diversity(images_base))
                    else:
                        logits = model(images_base)
                    if weight_np is None:
                        weight_np = torch.autograd.grad(
                            F.softmax(logits, dim=-1)*labels, features, grad_outputs=torch.ones_like(logits*labels))[0]
                    else:
                        weight_np += torch.autograd.grad(
                            F.softmax(logits, dim=-1)*labels, features, grad_outputs=torch.ones_like(logits*labels))[0]
                weight_np = -normalize(weight_np, 2)
            images_base = image_transform(torch.zeros_like(images))
            if "DI" in self.attack_method:
                _ = model(input_diversity(images_base))
            else:
                _ = model(images_base)
            base_feamap = features
            if "DI" in self.attack_method:
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

            if "PI" in self.attack_method:
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
