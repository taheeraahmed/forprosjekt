import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import transforms
from datasets import ModifiedNIH_Dataset, ChestXrayDatasetBinaryClass
import math
from sklearn.model_selection import train_test_split
from utils.get_images_list import get_images_list
import os
import pandas as pd

class MultiClassDataLoader:
    def __init__(self, data_path, test_mode, model_arg, batch_size, logger, train_frac=0.8):
        self.data_path = data_path
        self.test_mode = test_mode
        self.batch_size = batch_size
        self.model_arg = model_arg
        self.logger = logger
        self.train_frac = train_frac
        # TODO: These aren't passed forward to ModifiedNIH_Dataset BUG :))
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def list_directories(self):
        return [os.path.join(self.data_path, d, 'images') for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]

    def get_dataloaders(self):
        nih_img_dirs = self.list_directories()
        self.logger.info(f'Data directories: {nih_img_dirs}')
        dataset = ModifiedNIH_Dataset(imgpaths=nih_img_dirs, transforms=self.transforms, logger=self.logger, model_arg=self.model_arg)

        if len(dataset) == 0:
            self.logger.error('The dataset is empty.')
            raise ValueError("The dataset is empty. Please check your data source.")

        if self.test_mode:
            self.logger.warning('Using a subset of the dataset for testing')
            subset_size = 40
            indices = torch.randperm(len(dataset)).tolist()
            test_mode_subset_indices = indices[:subset_size]
            test_mode_subset_dataset = Subset(dataset, test_mode_subset_indices)

            # Adjust the sizes for splitting the subset
            test_size = int(0.8 * subset_size)  # 20% of the subset size
            validation_size = subset_size - test_size  # Rest of the subset for validation
            if validation_size <= 0:
                raise ValueError("Validation set has no data. Adjust the sizes.")

            test_dataset, validation_dataset = random_split(test_mode_subset_dataset, [test_size, validation_size])

            train_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        else:
            train_size = int(self.train_frac * len(dataset))
            self.logger.info(f'Using {train_size} fraction of dataset')
            validation_size = len(dataset) - train_size
            train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, validation_dataloader

class BinaryClassificationDataLoader:
    def __init__(self, data_path, test_mode, batch_size, logger, test_size=0.2, train_frac=0.25):
        self.data_path = data_path
        self.logger = logger
        self.dataframe = self._get_binary_classification_df(logger=logger)
        self.test_mode = test_mode
        self.batch_size = batch_size
        self.test_size = test_size
        self.train_frac = train_frac
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_binary_classification_df(self, logger):
        df = pd.read_csv(f'{self.data_path}/Data_Entry_2017.csv')
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
    
    def _prepare_dataframes(self):
        train_df, val_df = train_test_split(self.dataframe, test_size=self.test_size)

        if self.test_mode:
            test_df_size = 100
            val_df_size = math.floor(self.test_size * test_df_size)
            train_df = train_df.iloc[:test_df_size]
            val_df = val_df.iloc[:val_df_size]
        else: 
            train_df = train_df.sample(frac=self.train_frac)
            val_df = val_df.sample(frac=self.train_frac)

        self.logger.info(f"Train df shape: {train_df.shape}")
        self.logger.info(f"Validation df shape: {val_df.shape}")

        return train_df, val_df

    def get_dataloaders(self):
        train_df, val_df = self._prepare_dataframes()

        train_dataset = ChestXrayDatasetBinaryClass(df=train_df, transform=self.transform)
        val_dataset = ChestXrayDatasetBinaryClass(df=val_df, transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader