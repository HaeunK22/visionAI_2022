from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import random
import natsort

class Movie(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir

        self.image_names = [_ for _ in os.listdir(os.path.join(root_dir, 'madmax_1')) if _.endswith('jpg')]
        self.image_names = natsort.natsorted(self.image_names)
        self.resize = transforms.Resize((512,512))
        self.resize_1 = transforms.Resize((960,401))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)])
        self.num_images = len(self.image_names)
    def __len__(self):
        return 3

    def __getitem__(self, idx):

        image = self.image_names[idx*100]
        #next_frame = self.image_names[idx+1].split('_')[-1].split('.')[0]
        frames = image.split('_')[-1].split('.')[0]

        start = int(frames)
        end = int(frames) + 99
        #if end < int(next_frame):
        #    return torch.zeros(100), torch.zeros(100), torch.zeros(100), torch.zeros(100), torch.zeros(100), torch.zeros(100)


        image_path_start = os.path.join(self.root_dir, 'madmax_1/frame_'+str(start)+'.jpg')
        image_gray_start = Image.open(image_path_start).convert('L')
        image_rgb_start = Image.open(image_path_start)

        image_path_end = os.path.join(self.root_dir, 'madmax_1/frame_' + str(end-1) + '.jpg')
        image_gray_end = Image.open(image_path_end).convert('L')
        image_rgb_end = Image.open(image_path_end)

        image_rgb_start = np.array(self.resize(image_rgb_start))
        image_rgb_start = self.transform(image_rgb_start)

        image_gray_start = np.array(self.resize(image_gray_start))
        image_gray_start = self.transform(image_gray_start)

        image_rgb_end = np.array(self.resize(image_rgb_end))
        image_rgb_end = self.transform(image_rgb_end)

        image_gray_end = np.array(self.resize(image_gray_end))
        image_gray_end = self.transform(image_gray_end)

        image_rgb = torch.stack([image_rgb_start, image_rgb_end],dim=0)
        image_gray = torch.stack([image_gray_start, image_gray_end], dim=0)

        frms = []
        gfrms = []
        gtfrms=[]
        imgs = []
        gimgs = []

        for i in range(start, end, 10):
            img = Image.open(os.path.join(self.root_dir, 'madmax_1/frame_' + str(i) + '.jpg'))
            img = np.array(self.resize(img))
            img = self.transform(img)
            imgs.append(img)
            gimg = Image.open(os.path.join(self.root_dir, 'madmax_2/frame_' + str(i) + '.jpg')).convert('L')
            gimg = np.array(self.resize(gimg))
            gimg = self.transform(gimg)
            gimgs.append(gimg)

        for i in range(start, end):
            frm = Image.open(os.path.join(self.root_dir, 'madmax_1/frame_'+str(i)+'.jpg' ))
            frm = np.array(frm)
            frm = self.transform(frm)
            frms.append(frm)
            gfrm = Image.open(os.path.join(self.root_dir, 'madmax_1/frame_' + str(i) + '.jpg')).convert('L')
            gfrm = np.array(gfrm)
            gfrm = self.transform(gfrm)
            gfrms.append(gfrm)
            gtfrm = Image.open(os.path.join(self.root_dir, 'madmax_2/frame_' + str(i) + '.jpg')).convert('L')
            gtfrm = np.array(gtfrm)
            gtfrm = self.transform(gtfrm)
            gtfrms.append(gtfrm)
        frms = torch.stack(frms,dim=0)
        gfrms = torch.stack(gfrms,dim=0)
        gtfrms = torch.stack(gtfrms,dim=0)
        imgs = torch.stack(imgs,dim=0)
        gimgs = torch.stack(gimgs,dim=0)

        return imgs, gimgs, image, frms, gfrms, gtfrms

