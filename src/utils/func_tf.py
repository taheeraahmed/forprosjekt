import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
import pyfiglet
import logging
import os
from datetime import datetime
from utils.create_dir import create_directory_if_not_exists
from utils.check_gpu import check_gpu
from datetime import datetime, timedelta
import time

def set_up_tf(args):
    result = pyfiglet.figlet_format("Tensorflow", font = "slant")  
    print(result) 
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if args.class_imbalance:
        job_name = f'{args.date}-{args.model}-imbalance-tf'
    else:
        job_name = f'{args.date}-{args.model}-tf'

    LOG_DIR = f'/cluster/home/taheeraa/code/forprosjekt/output/tf/{job_name}'
    create_directory_if_not_exists(LOG_DIR)
    
    LOG_FILE = f"{LOG_DIR}/log_file.txt"

    logging.basicConfig(level=logging.INFO, 
                format='[%(levelname)s] %(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(LOG_FILE),
                    logging.StreamHandler()
                ])
    logger = logging.getLogger()
    logger.info(f'Running: {args.model}')
    logger.info(f'Root directory of project: {project_root}')
    logger.info(f'Log directory: {LOG_DIR}')

    logger.info(f'Batch size: {args.BATCH_SIZE}')
    logger.info(f'Shuffle buffer size: {args.SHUFFLE_BUFFER_SIZE}')
    logger.info(f'Epochs: {args.EPOCHS}')


    check_gpu(logger)
    logger.info('Set-up completed')

    return logger, LOG_DIR



def save_plot(history, history_metric, val_history_metric, metric_name, file_name):
    plt.figure()
    sns.lineplot(data=history.history[history_metric], label='Train')
    sns.lineplot(data=history.history[val_history_metric], label='Val')
    plt.title(f'Experiment #1 {metric_name}')
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(file_name)
    plt.close()

def get_df(data_path, class_imbalance=False):
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

    df['No Finding'] = df[diseases].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)
    
    if class_imbalance:
        # Undersampling the majority class
        min_class_count = min(df[diseases + ['No Finding']].sum())
        df = pd.concat([
            df[df[label] == 1].sample(min_class_count, replace=True) for label in diseases + ['No Finding']
        ]).sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
    
    df = df.drop('No Finding', axis=1)
    
    # used for handling data leak
    patient_ids = df['Patient ID'].unique()

    # split the patient IDs into train, validation, and test sets
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=0)

    train_df = df[df['Patient ID'].isin(train_ids)]
    val_df = df[df['Patient ID'].isin(val_ids)]

    # drop the 'Patient ID' column
    train_df = train_df.drop('Patient ID', axis=1).reset_index(drop=True)
    val_df = val_df.drop('Patient ID', axis=1).reset_index(drop=True)

    return train_df, val_df

class ChestXray14TFDataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
        self.labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
               'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
               'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

    def _parse_function(self, filename, labels):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [224, 224])
        # Ensure the image is in the range [0, 1]
        image_normalized = image_resized / 255.0
        # Apply other transformations as needed
        return image_normalized, labels

    def get_dataset(self):
        filenames = self.dataframe.iloc[:, 0].values
        labels = self.dataframe.iloc[:, 1:].values.astype('float32')
        labels = np.array(labels)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(self._parse_function)
        # Add dataset.shuffle, dataset.batch, etc. as needed
        return dataset
