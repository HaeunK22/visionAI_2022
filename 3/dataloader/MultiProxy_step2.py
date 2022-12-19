from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random


class MultiProxy_step2(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir+'/train')
        self.objects = os.listdir(os.path.join(root_dir+'/train', self.classes[0]))
        self.resize = transforms.Resize((512, 512))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)])
        self.split = split
        if split == 'train':
            building = [_ for _ in os.listdir(self.root_dir + '/train/Input/building') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            nonhuman = [_ for _ in os.listdir(self.root_dir + '/train/Input/nonhuman') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            human = [_ for _ in os.listdir(self.root_dir + '/train/Input/human') if _.endswith('.jpg') or _.endswith('.JPG')]
            nature = [_ for _ in os.listdir(self.root_dir + '/train/Input/nature') if
                      _.endswith('.jpg') or _.endswith('.JPG')]
            self.image_names = np.concatenate([nonhuman, human, building, nature])


        if split == 'val':
            building = [_ for _ in os.listdir(self.root_dir + '/val/Input/building') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            nonhuman = [_ for _ in os.listdir(self.root_dir + '/val/Input/nonhuman') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            human = [_ for _ in os.listdir(self.root_dir + '/val/Input/human') if _.endswith('.jpg') or _.endswith('.JPG')]
            nature = [_ for _ in os.listdir(self.root_dir + '/val/Input/nature') if
                      _.endswith('.jpg') or _.endswith('.JPG')]
            self.image_names = np.concatenate([nonhuman, human, building, nature])


        if split == 'test':
            building = [_ for _ in os.listdir(self.root_dir + '/test/Input/building') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            nonhuman = [_ for _ in os.listdir(self.root_dir + '/test/Input/nonhuman') if
                        _.endswith('.jpg') or _.endswith('.JPG')]
            human = [_ for _ in os.listdir(self.root_dir + '/test/Input/human') if _.endswith('.jpg') or _.endswith('.JPG')]
            nature = [_ for _ in os.listdir(self.root_dir + '/test/Input/nature') if
                      _.endswith('.jpg') or _.endswith('.JPG')]
            self.image_names = np.concatenate([nonhuman, human, building, nature])


        self.n_bu = len(building)
        self.n_hu = len(human)
        self.n_nh = len(nonhuman)
        self.n_na = len(nature)

        self.num_classes = len(self.classes)
        self.num_images = len(self.image_names)

    def __len__(self):
        return self.num_images * 3

    def __getitem__(self, idx):
        if self.split == 'train':
            root_dir = self.root_dir + '/train'
        elif self.split == 'val':
            root_dir = self.root_dir + '/val'
        else:
            root_dir = self.root_dir + '/test'

        style_l = idx // (self.num_images)
        if style_l == 2:
            style_l = 3
        if (idx % self.num_images) < self.n_nh:
            objects_l = 0
        elif idx % self.num_images < (self.n_nh + self.n_hu):
            objects_l = 1
        elif idx % self.num_images < (self.n_nh + self.n_hu + self.n_bu):
            objects_l = 2
        else:
            objects_l = 3

        images_l = idx % self.num_images

        image_path = os.path.join(root_dir, self.classes[2], self.objects[objects_l], self.image_names[images_l])
        gt_path = os.path.join(root_dir, self.classes[style_l], self.objects[objects_l], self.image_names[images_l])

        rgb_img = Image.open(image_path).convert('RGB')
        rgb_img = np.array(self.resize(rgb_img))
        rgb_img = self.transform(rgb_img)

        gray_img = Image.open(image_path).convert('L')
        gray_img = np.array(self.resize(gray_img))
        gray_img = self.transform(gray_img)

        gt_img = Image.open(gt_path).convert('L')
        gt_img = np.array(self.resize(gt_img))
        gt_img = self.transform(gt_img)
        return rgb_img, gray_img, gt_img, self.image_names[images_l], style_l

