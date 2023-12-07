import os
from PIL import Image
import torch
import PIL.Image as Image

from torch.utils.data import Dataset

class ChestXrayMultiClassDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        labels = self.dataframe.iloc[idx, 1:].to_numpy()
        labels = torch.from_numpy(labels.astype('float32'))

        if self.transform:
            image = self.transform(image)

        return {"img": image, "lab": labels}

