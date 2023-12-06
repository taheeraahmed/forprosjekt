import os
from PIL import Image
import torch
from skimage.io import imread
import numpy as np
import PIL.Image as Image
from torch import Tensor

from torch.utils.data import Dataset
from torchxrayvision.datasets import NIH_Dataset, apply_transforms
from torchxrayvision.utils import normalize
from torchvision import transforms
import torchvision
import torchxrayvision as xrv
import sys
class ModifiedNIH_Dataset(NIH_Dataset):
    def __init__(self, imgpaths, transforms, logger = None, model_arg=None, *args, **kwargs):
        self.imgpaths = imgpaths  # Set imgpaths attribute before calling super
        self.logger = logger
        self.transforms = transforms
        self.model_arg = model_arg

        if 'imgpath' not in kwargs:
            kwargs['imgpath'] = imgpaths[0] if imgpaths else None

        super(ModifiedNIH_Dataset, self).__init__(*args, **kwargs)

    def check_paths_exist(self):
        # Override the method to check each path in imgpaths
        for path in self.imgpaths:
            if not os.path.isdir(path):
                raise Exception(f"{path} must be a directory")

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]

        # Determine which directory the image is in
        for img_path in self.imgpaths:
            full_img_path = os.path.join(img_path, imgid)
            if os.path.exists(full_img_path):
                break
        # TODO FIX
        

        if self.model_arg=="densenet-pretrained-xray-multi-class":
            img = imread(full_img_path)
            sample["img"] = normalize(img, maxval=255, reshape=True)

            if self.pathology_masks:
                sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

            # applying transforms
            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
            sample = apply_transforms(sample, transform)
            sample = apply_transforms(sample, self.data_aug)
            
            return sample

        elif self.model_arg=="vit-imagenet-multi-class":
            img = Image.open(full_img_path).convert('RGB') 
            debug_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]) 
            img_transformed = debug_transforms(img)

            if img_transformed.shape != (3,224,224):
                raise ValueError(f"Unexpected size {img.shapee}") 
            sample = {"img": img_transformed, "lab": self.labels[idx]}
            
            if self.pathology_masks:
                sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

            img = sample['img']
            if type(img) != Tensor:
                raise TypeError(f"Unexpected type {type(img)}") 
            return sample
            
        else:
            self.logger.error('Invalid moodel_arg')
            sys.exit(1)

        
    
    def __len__(self):
        return super(ModifiedNIH_Dataset, self).__len__()

class ChestXrayDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (DataFrame): DataFrame with image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.df.iloc[idx]['image_files'])
        image = Image.open(img_name)
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        
        label = self.df.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.float32)
        return image, label