import os
import pandas as pd
from PIL import Image
import torch

from torch.utils.data import Dataset
from utils.get_images_list import get_images_list

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

# Your transform

def get_binary_classification_df(logger, data_path):
    df = pd.read_csv(f'{data_path}/Data_Entry_2017.csv')
    # Add the image paths to the dataframe
    df['Image Files List'] = get_images_list(logger)
    # Create a new dataframe to store the image paths, labels, and patient IDs
    df = df[['Image Files List', 'Finding Labels', 'Patient ID']]

    # Make a list of all the labels
    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                'Pneumonia', 'Pneumothorax']

    # For each label, make a new column and
    # assign 1 if the disease is present and 0 if the disease is absent
    for disease in diseases:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
        
    disease_mapping = {
        'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Edema': 3,
        'Effusion': 4, 'Emphysema': 5, 'Fibrosis': 6, 'Hernia': 7,
        'Infiltration': 8, 'Mass': 9, 'Nodule': 10, 'Pleural_Thickening': 11,
        'Pneumonia': 12, 'Pneumothorax': 13
    }

    # Function to get the first disease and map it
    def map_disease_to_number(label):
        first_disease = label.split('|')[0]
        return disease_mapping.get(first_disease, -1)  # Returns -1 if disease not found

    # Apply the function to create the new column
    df['image_class'] = df['Finding Labels'].apply(map_disease_to_number)
    df['finding_indicator'] = df['Finding Labels'].apply(lambda x: 0 if x == 'No Finding' else 1)
    df = df.rename(columns={'Image Files List': 'image_files'})
    df = df[['image_files', 'finding_indicator']]
    logger.info('Created dataframe')
    return df


