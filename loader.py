import os
from PIL import Image
from torch.utils import data
import pandas as pd
from torchvision import transforms as T
import torch
import torch.nn as nn

class ImageNet(data.Dataset):
    

    def __init__(self, dir, csv_path, transforms=None, num_images=None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms
        self.num_images = num_images


    def __getitem__(self, index):
        if self.num_images is not None and index >= self.num_images:
            raise StopIteration
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, ImageID, Truelabel


    def __len__(self):
        if self.num_images is None:
            return len(self.csv)
        else:
            return min(len(self.csv), self.num_images)
        


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

class TfNormalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(TfNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class Permute(nn.Module):
    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        return input[:, self.permutation]
