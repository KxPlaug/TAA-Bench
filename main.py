from attacks.bim import BIM
from attacks.danaa import DANAA
from attacks.difgsm import DIFGSM
from attacks.igattack import IGAttack
from attacks.mifgsm import MIFGSM
from attacks.naa import NAA
from attacks.sinifgsm import SINIFGSM
from attacks.ssa import SSA

from omegaconf import OmegaConf
import pretrainedmodels
import os
import torch
from torchvision import transforms as T
from loader import ImageNet,Normalize,TfNormalize
import torch.nn as nn
import copy
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as T
from functools import partial
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
)
import sys

class ReturnFirst(nn.Module):
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = x[:,1:]
        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    
setup_seed(2023)

def get_model(net_name, model_dir):
    """Load converted model"""
    if isinstance(net_name, str):
        model_path = os.path.join(model_dir, net_name + '.npy')

        if net_name == 'tf_inception_v3':
            net = tf_inception_v3
        elif net_name == 'tf_inception_v4':
            net = tf_inception_v4
        elif net_name == 'tf_resnet_v2_50':
            net = tf_resnet_v2_50
        elif net_name == 'tf_resnet_v2_101':
            net = tf_resnet_v2_101
        elif net_name == 'tf_resnet_v2_152':
            net = tf_resnet_v2_152
        elif net_name == 'tf_inc_res_v2':
            net = tf_inc_res_v2
        elif net_name == 'tf_adv_inception_v3':
            net = tf_adv_inception_v3
        elif net_name == 'tf_ens3_adv_inc_v3':
            net = tf_ens3_adv_inc_v3
        elif net_name == 'tf_ens4_adv_inc_v3':
            net = tf_ens4_adv_inc_v3
        elif net_name == 'tf_ens_adv_inc_res_v2':
            net = tf_ens_adv_inc_res_v2
        else:
            print('Wrong model name!')
        rf = ReturnFirst().eval().cuda()
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),
            rf
        )
        return model
    else:
        return net_name

def verify(model_name, path, adv_dir, input_csv, batch_size=10,num_images=1000):

    model = get_model(model_name, path)

    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]), num_images=num_images)
    data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(-1) != (gt)).detach().sum().cpu()

    if isinstance(model_name, str):
        print(model_name + '  acu = {:.2%}'.format(sum / num_images))
    else:
        print("Torch Model" + '  acu = {:.2%}'.format(sum / num_images))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="configs/template.yaml")
args = parser.parse_args()
config = OmegaConf.load(args.config)
PREFIX = os.path.basename(args.config)[:-5]
METHOD = eval(config.attack.method)
BATCH_SIZE = config.data.batch_size
EPS = config.attack.eps / 255
NUM_IMAGES = config.data.num_images
STEPS = config.attack.steps
MODEL = config.model.name
TRAIN_STEPS = config.model.train_config.train_steps
LEARING_RATE = config.model.train_config.learning_rate
SAVE_STEPS = config.attack.save_steps
ALPHA = EPS / STEPS
OUTPUT_DIR = "output/"+PREFIX+"/"

sys.stdout = open(f"logs/{PREFIX}.log", "w")
def partial_save_image(names):
    return partial(save_image,names=names)

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

dataset = ImageNet("dataset/images","dataset/images.csv",transforms=transforms,num_images=NUM_IMAGES)


dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

model = eval("pretrainedmodels."+MODEL)

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

model = torch.nn.Sequential(Normalize(mean, std),model(num_classes=1000, pretrained='imagenet').eval().to(device)).eval()

attack = METHOD(copy.deepcopy(model),eps=EPS,alpha=ALPHA,steps=STEPS)
success_rates = []
pbar = tqdm(total=len(dataloader))
for images, images_ID,  gt_cpu in dataloader:
    images = images.to(device)
    gt = gt_cpu.to(device)
    adv_images,success_step = attack(images,gt,partial_save_image(images_ID),save_steps=SAVE_STEPS,output_dir=OUTPUT_DIR)
    success_rates.append((model(adv_images).argmax(dim=-1) != gt).float().mean().item())
    pbar.set_description("success rate: %.4f" % np.mean(success_rates))
    pbar.update(1)

pbar.close()
print("success rate: %.4f" % np.mean(success_rates))

MAPPER = {
    'inceptionv3': 'tf_inception_v3',
    'inceptionv4': 'tf_inception_v4',
    'inceptionresnetv2': 'tf_inc_res_v2',
    'resnet50': 'tf_resnet_v2_50',
    'resnet101': 'tf_resnet_v2_101',
    'resnet152': 'tf_resnet_v2_152',
    'ens3_adv_inceptionv3': 'tf_ens3_adv_inc_v3',
    'ens4_adv_inceptionv3': 'tf_ens4_adv_inc_v3',
    'ens_adv_inceptionresnetv2': 'tf_ens_adv_inc_res_v2'
}

# model_names = ['tf_inception_v3','tf_inception_v4','tf_inc_res_v2','tf_resnet_v2_50','tf_resnet_v2_101','tf_resnet_v2_152','tf_ens3_adv_inc_v3','tf_ens4_adv_inc_v3','tf_ens_adv_inc_res_v2']

model_names = [MAPPER[transformed_model_name] for transformed_model_name in config.transfer.models]

models_path = './models/'

for model_name in model_names:
    verify(model_name, models_path, OUTPUT_DIR, 'dataset/images.csv', batch_size=BATCH_SIZE, num_images=NUM_IMAGES)

# for step in SAVE_STEPS:
#     print(f"verify for step {step}")

#     for model_name in model_names:
#         verify(model_name, models_path, OUTPUT_DIR[:-1] + f"_{step}/", 'dataset/images.csv', batch_size=BATCH_SIZE, num_images=NUM_IMAGES)
#     verify(model, models_path, OUTPUT_DIR[:-1] + f"_{step}/", 'dataset/images.csv', batch_size=BATCH_SIZE, num_images=NUM_IMAGES)