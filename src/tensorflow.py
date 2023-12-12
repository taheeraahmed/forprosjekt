import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from utils.set_up import str_to_bool, set_up_tf
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from keras import layers, models
from tfswin import SwinTransformerTiny224, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.metrics import AUC
import tensorflow_addons as tfa
from tensorflow.keras.applications import DenseNet121
import pandas as pd

DATA_PATH = '/cluster/home/taheeraa/datasets/chestxray-14'
BATCH_SIZE = 32  # You can adjust this according to your system's capability
SHUFFLE_BUFFER_SIZE = 1000  # Adjust as needed
EPOCHS = 20

def train(args):
    args.BATCH_SIZE = BATCH_SIZE
    args.SHUFFLE_BUFFER_SIZE = SHUFFLE_BUFFER_SIZE
    args.EPOCHS = EPOCHS

    logger, log_dir = set_up_tf(args)
    
    train_df, val_df = get_df(DATA_PATH, class_imbalance=args.class_imbalance)
    logger.info(f'Shapes: {train_df.shape}, {val_df.shape}')
    train_tf_dataset = ChestXray14TFDataset(train_df)
    val_tf_dataset = ChestXray14TFDataset(val_df)
    num_classes = len(train_tf_dataset.labels)
    train_tf_dataset = train_tf_dataset.get_dataset()
    val_tf_dataset = val_tf_dataset.get_dataset()
    train_tf_dataset = train_tf_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    
    if args.model == 'swin':
        logger.info('Using swin')
        inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
        outputs = layers.Lambda(preprocess_input)(inputs)
        x = SwinTransformerTiny224(include_top=False)(outputs) 
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=predictions) 

        for layer in model.layers[:-3]:
            layer.trainable = False
        
    elif args.model == 'densenet':
        logger.info('Using densenet')
        base_model = DenseNet121(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=[AUC(), tfa.metrics.F1Score(num_classes=num_classes, average='macro')])

    history = model.fit(train_tf_dataset,
                        epochs=EPOCHS,
                        validation_data=val_tf_dataset)
    
    hist_df = pd.DataFrame(history.history)

    hist_df.to_csv(f'{log_dir}/{args.model}-{args.class_imbalance}.csv', index=False)
    logger.info(f"Saved model history to model_history/{args.model}-{args.class_imbalance}.csv")

    save_plot(history, 'auc', 'val_auc', 'AUC', f'{log_dir}/{args.model}-{args.class_imbalance}-auc-plot.png')
    save_plot(history, 'loss', 'val_loss', 'Loss', f'{log_dir}/{args.model}-{args.class_imbalance}-loss-plot.png')
    save_plot(history, 'f1_score', 'val_f1_score', 'F1 Score', f'{log_dir}/{args.model}-{args.class_imbalance}-f1-score-plot.png')

if __name__ == "__main__":
    model_choices = ['densenet','swin']

    parser = argparse.ArgumentParser(description="Arguments for training with pytorch")
    parser.add_argument("-m", "--model", choices=model_choices, help="Model to run", required=True)
    parser.add_argument("-im", "--class_imbalance", help="Handle class imbalance", required=False, default=False)
    parser.add_argument("-date", "--date", help="Start time", type=str, required=True)

    args = parser.parse_args()
    args.class_imbalance = str_to_bool(args.class_imbalance)
    train(args)


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
