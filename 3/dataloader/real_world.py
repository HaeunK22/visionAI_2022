from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random


class Real(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.image_names = [_ for _ in os.listdir(root_dir) if _.endswith('jpg') or _.endswith('JPG')]

        self.resize = transforms.Resize((512,512))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)])
        self.image_names = np.array(self.image_names)

        self.num_images = len(self.image_names)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        image_path = os.path.join(self.root_dir, self.image_names[idx])
        if self.image_names[idx].endswith('.JPG'):
            k = self.image_names[idx].replace('.JPG', '.jpg')
        else:
            k = self.image_names[idx]
        rgb_img = Image.open(image_path).convert('RGB')
        rgb_img = np.array(self.resize(rgb_img))
        rgb_img = self.transform(rgb_img)

        rgb_img_ = Image.open(image_path).convert('RGB')
        rgb_img_ = np.array(rgb_img_)
        rgb_img_ = self.transform(rgb_img_)

        gray_img = Image.open(image_path).convert('L')
        gray_img = np.array(gray_img)
        gray_img = self.transform(gray_img)

        return rgb_img, gray_img, self.image_names[idx], rgb_img_

