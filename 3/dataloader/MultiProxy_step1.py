from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random
from itertools import chain

class MultiProxy(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir+'/full'
        self.classes = os.listdir(self.root_dir)
        self.objects = os.listdir(os.path.join(self.root_dir, self.classes[0]))
        self.image_names = [[_ for _ in os.listdir(os.path.join(self.root_dir, self.classes[0],i )) if _.endswith('.jpg')] for i in self.objects]

        self.resize = transforms.Resize((224,224))


        if split == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
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

        if split == 'val':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float)])
            building = [_ for _ in os.listdir(self.root_dir + '/Input/building') if
                        _.endswith('.jpg') or _.endswith('.JPG')][-250:]
            nonhuman = [_ for _ in os.listdir(self.root_dir + '/Input/nonhuman') if
                        _.endswith('.jpg') or _.endswith('.JPG')][-250:]
            human = [_ for _ in os.listdir(self.root_dir + '/Input/human') if
                     _.endswith('.jpg') or _.endswith('.JPG')][-250:]
            nature = [_ for _ in os.listdir(self.root_dir + '/Input/nature') if
                      _.endswith('.jpg') or _.endswith('.JPG')][-250:]
            self.image_names = np.concatenate([nonhuman, human, building, nature])

        self.n_bu = len(building)
        self.n_hu = len(human)
        self.n_nh = len(nonhuman)
        self.n_na = len(nature)

        self.num_classes = len(self.classes)
        self.num_images = len(self.image_names)

    def __len__(self):
        return self.num_images * self.num_classes

    def __getitem__(self, idx):

        style_l = idx // (self.num_images)

        if (idx % self.num_images) < self.n_nh:
            objects_l = 0
        elif idx % self.num_images < (self.n_nh + self.n_hu):
            objects_l = 1
        elif idx % self.num_images < (self.n_nh + self.n_hu + self.n_bu):
            objects_l = 2
        else:
            objects_l = 3

        images_l = idx % self.num_images

        path_n = self.classes[style_l]
        objects_n = self.objects[objects_l]
        image_n = self.image_names[images_l]

        image_path = os.path.join(self.root_dir, path_n, objects_n, image_n)
        image = Image.open(image_path).convert('L')
        image = np.array(self.resize(image))
        image = self.transform(image)
        return image, style_l, objects_l, style_l * len(self.objects) + objects_l

