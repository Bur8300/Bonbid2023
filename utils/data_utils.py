import torch
from medpy.io import load
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from torch.utils.data import Dataset
from torchvision import transforms

def return_images(image_paths):
    arr = []
    for path in image_paths:
        imgs, _ = load(path)
        for i in range(imgs.shape[-1]):
            arr.append(imgs[:,:,i])
    return arr
class Bonbid2023(Dataset):
    '''
    Custom PyTorch Dataset class to load and preprocess images and their corresponding segmentation masks.
    
    Args:
    - data_path (str): The root directory of the dataset.
    - mode (str): The mode in which the dataset is used, either 'train' or 'test'.
    
    Attributes:
    - image_paths (list): Sorted list of file paths for images.
    - mask_paths (list): Sorted list of file paths for masks.
    - transform (Compose): Transformations to apply to the images (convert to tensor and resize).
    - mask_transform (Compose): Transformations to apply to the masks (convert to tensor and resize).
    - augment (bool): Whether to apply data augmentation (only for training mode).
    - augmentation_transforms (Compose): Augmentation transformations (horizontal flip, vertical flip, color jitter).
    '''

    def __init__(self, mode='train') -> None:
        # Load and sort image and mask file paths
        self.image_paths = sorted(glob.glob(os.path.join("BONBID2023_Train/1ADC_ss", '*_ss*.mha')))
        self.mask_paths  = sorted(glob.glob(os.path.join("BONBID2023_Train/3LABEL", '*_lesion*.mha')))
        self.images = return_images(self.image_paths)
        self.masks = return_images(self.mask_paths)

        # Ensure the number of images and masks match
        assert len(self.images) == len(self.masks)

        # Define transformations for images and masks: convert to tensor and resize to 256x256
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128,128)),            #Normalize images
            transforms.Normalize(0.5,0.5)
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128,128))
        ])

        # Determine if augmentation should be applied (only in 'train' mode)
        self.augment = True if mode == "train" else False
        self.augment = False

        # Define augmentation transformations for training
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter()
        ])
        
    def __len__(self):
        # Return the total number of samples
        return len(self.images)

    def __getitem__(self, index):
        '''
        Load and preprocess an image and its corresponding mask at the given index.
        
        Args:
        - index (int): Index of the sample to fetch.
        
        Returns:
        - img (Tensor): Transformed image tensor.
        - mask (Tensor): Transformed mask tensor.
        '''
        # Load the image and mask using OpenCV (image in grayscale, mask with unchanged properties)
        img = self.images[index]
        mask = self.masks[index]
        # Apply transformations to the image and mask
        img = self.transform(img)
        mask = self.mask_transform(mask)

        # Apply data augmentation if enabled
        if self.augment:
            # Set random seed to ensure consistency between image and mask transformations
            torch.manual_seed(42)
            img = self.augmentation_transforms(img)
            mask = self.augmentation_transforms(mask)
        return img, mask
