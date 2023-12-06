import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob

def handle_class_imbalance_df(data_path, logger):
    df = pd.read_csv(f'{data_path}/Data_Entry_2017.csv')
    image_paths = []
    for i in range(1, 13):
        folder_name = f'{data_path}/images_{i:03}'
        files_in_subfolder = glob.glob(f'{folder_name}/images/*')
        image_paths.extend(files_in_subfolder)
    assert len(image_paths) == 112120, f"Expected 112120 images, but found {len(image_paths)}"
    df['Image Path'] = image_paths
    df = df[['Image Path', 'Finding Labels', 'Patient ID']]

    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                'Pneumonia', 'Pneumothorax']
    
    # one-hot encoding disease and dropping finding labels
    for disease in diseases:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
    df = df.drop('Finding Labels', axis=1)

    # used for handling data leak
    patient_ids = df['Patient ID'].unique()

    # split the patient IDs into train, validation, and test sets
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=0)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.26, random_state=0) # 0.26 x 0.8 ~= 0.2

    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]
    test_df = df[df['Patient ID'].isin(test_ids)]

    # check the shapes of the dataframes
    logger.info(f'train_df shape: {train_df.shape}, val_df shape: {val_df.shape}, test_df shape:, {test_df.shape}')
    # check the ratios of the dataframes as we split based on patient IDs, not individual images
    logger.info(f'train_df ratio: {round(len(train_df) / len(df), 3)}, val_df ratio: {round(len(val_df) / len(df), 3)}, test_df ratio: {round(len(test_df) / len(df), 3)}')

    # drop the 'Patient ID' column
    train_df = train_df.drop('Patient ID', axis=1).reset_index(drop=True)
    val_df = val_df.drop('Patient ID', axis=1).reset_index(drop=True)
    test_df = test_df.drop('Patient ID', axis=1).reset_index(drop=True)
    return train_df, val_df, test_df 

def get_class_weights(train_df):
    class_weights = []

    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                'Pneumonia', 'Pneumothorax']

    for i, disease in enumerate(diseases):
        # count the number of positive and negative instances for this disease
        n_positive = np.sum(train_df[disease])
        n_negative = len(train_df) - n_positive

        # compute the weight for positive instances and the weight for negative instances
        weight_for_positive = (1 / n_positive) * (len(train_df) / 2.0)
        weight_for_negative = (1 / n_negative) * (len(train_df) / 2.0)

        class_weights.append({0: weight_for_negative, 1: weight_for_positive})