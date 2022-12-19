from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random


class test(Dataset):
    def __init__(self, root_dir, style):
        self.root_dir = root_dir +'/test'
        self.classes = ['Input', 'Expert_'+style]
        self.objects = os.listdir(os.path.join(self.root_dir, self.classes[0]))
        self.image_names = [[_ for _ in os.listdir(os.path.join(self.root_dir, self.classes[0],i )) if _.endswith('.jpg')] for i in self.objects]

        self.resize = transforms.Resize((512,512))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)])

        building = [_ for _ in os.listdir(self.root_dir + '/Input/building') if
                    _.endswith('.jpg') or _.endswith('.JPG')]
        nonhuman = [_ for _ in os.listdir(self.root_dir + '/Input/nonhuman') if
                    _.endswith('.jpg') or _.endswith('.JPG')]
        human = [_ for _ in os.listdir(self.root_dir + '/Input/human') if
                 _.endswith('.jpg') or _.endswith('.JPG')]
        nature = [_ for _ in os.listdir(self.root_dir + '/Input/nature') if
                  _.endswith('.jpg') or _.endswith('.JPG')]
        self.image_names = np.concatenate([nonhuman, human, building, nature])

        self.num_images = len(self.image_names)
        self.n_bu = len(building)
        self.n_hu = len(human)
        self.n_nh = len(nonhuman)
        self.n_na = len(nature)
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        if idx < self.n_nh:
            objects_l = 0
        elif idx < (self.n_nh + self.n_hu):
            objects_l = 1
        elif idx < (self.n_nh + self.n_hu + self.n_bu):
            objects_l = 2
        else:
            objects_l = 3

        obj_name = self.objects[objects_l]
        image_path = os.path.join(self.root_dir, self.classes[0], obj_name, self.image_names[idx])
        gt_path = os.path.join(self.root_dir, self.classes[1], obj_name, self.image_names[idx])

        rgb_img = Image.open(image_path).convert('RGB')
        rgb_img = np.array(self.resize(rgb_img))
        rgb_img = self.transform(rgb_img)

        gray_img = Image.open(image_path).convert('L')
        gray_img = np.array(self.resize(gray_img))
        gray_img = self.transform(gray_img)

        gt_img = Image.open(gt_path).convert('L')
        gt_img = np.array(self.resize(gt_img))
        gt_img = self.transform(gt_img)
        return rgb_img, gray_img, gt_img, self.image_names[idx]

