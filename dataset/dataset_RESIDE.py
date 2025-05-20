import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

def get_root_name(filename):
    base_name = os.path.splitext(filename)[0]
    underscore_count = base_name.count('_')
    if underscore_count > 1:
        parts = base_name.rsplit('_', 1)
        if parts[1].isdigit():
            return parts[0]
    return base_name

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, phase, img_size=(256,256), evaluate=False, suffix =''):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.evaluate = evaluate
        
        self.img_size = img_size
        self.hazy_dir = os.path.join(root_dir, f"{phase}A" + suffix)
        self.clear_dir = os.path.join(root_dir, f"{phase}B" + suffix)

        self.hazy_files = sorted([
            f for f in os.listdir(self.hazy_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ])

        self.clear_files = sorted([
            f for f in os.listdir(self.clear_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ])

        self.clear_dict = {
            get_root_name(f): f for f in self.clear_files
        }

        self.final_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        self.augment_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(lambda img: img.rotate(random.choice([0, 90, 180, 270]))),
            transforms.ToTensor()
        ])
        

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy_name = self.hazy_files[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        root_name = get_root_name(hazy_name)

        if root_name not in self.clear_dict:
            raise ValueError(f"No matching clear image found for root='{root_name}'.")

        gt_name = self.clear_dict[root_name]
        gt_path = os.path.join(self.clear_dir, gt_name)

        hazy_img = Image.open(hazy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        
        output = (
            self.final_transform(hazy_img),
            self.final_transform(gt_img)
        )

        if self.evaluate:
          output = output + (hazy_name,)

        return output
