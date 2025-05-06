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
    def __init__(self, root_dir, phase, steps=500, mix=False):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.steps = steps
        self.mix = mix

        self.cloudy_dir = os.path.join(root_dir, f"{phase}A")
        self.clear_dir = os.path.join(root_dir, f"{phase}B")

        self.cloudy_files = sorted([
            f for f in os.listdir(self.cloudy_dir)
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
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.augment_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(lambda img: img.rotate(random.choice([0, 90, 180, 270]))),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.cloudy_files)

    def __getitem__(self, idx):
        cloudy_name = self.cloudy_files[idx]
        cloudy_path = os.path.join(self.cloudy_dir, cloudy_name)
        root_name = get_root_name(cloudy_name)

        if root_name not in self.clear_dict:
            raise ValueError(f"No matching clear image found for root='{root_name}'.")

        if self.mix:
          rand_idx = random.randint(0,len(self.cloudy_files)-1)
          root_name_mix = get_root_name(self.cloudy_files[rand_idx])
          clear_name = self.clear_dict[root_name_mix]
        else:
          clear_name = self.clear_dict[root_name]
        clear_path = os.path.join(self.clear_dir, clear_name)

        gt_name = self.clear_dict[root_name]
        gt_path = os.path.join(self.clear_dir, gt_name)

        cloudy_img = Image.open(cloudy_path).convert("RGB")
        clear_img = Image.open(clear_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        return (
            self.final_transform(cloudy_img),
            self.final_transform(clear_img),
            self.final_transform(gt_img)
        )
