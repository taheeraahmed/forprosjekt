import os
from PIL import Image
import torch
import PIL.Image as Image

from torch.utils.data import Dataset


class ChestXrayDatasetBinaryClass(Dataset):
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