import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from utils.set_up import str_to_bool
import argparse
from keras import layers, models
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.metrics import AUC
from keras.callbacks import EarlyStopping
from keras import layers, models
from tfswin import SwinTransformerTiny224, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.metrics import AUC
import tensorflow_addons as tfa

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import DenseNet121
import pandas as pd

from utils.func_tf import explain_model, get_df, ChestXray14TFDataset, save_plot, set_up_tf

DATA_PATH = '/cluster/home/taheeraa/datasets/chestxray-14'
SHUFFLE_BUFFER_SIZE = 1000  # Adjust as needed



def train(args):
    args.SHUFFLE_BUFFER_SIZE = SHUFFLE_BUFFER_SIZE

    logger, log_dir = set_up_tf(args)
    
    train_df, val_df = get_df(DATA_PATH, class_imbalance=args.class_imbalance)
    logger.info(f'Shapes: {train_df.shape}, {val_df.shape}')
    train_tf_dataset = ChestXray14TFDataset(train_df)
    val_tf_dataset = ChestXray14TFDataset(val_df)
    num_classes = len(train_tf_dataset.labels)
    train_tf_dataset = train_tf_dataset.get_dataset()
    val_tf_dataset = val_tf_dataset.get_dataset()
    train_tf_dataset = train_tf_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_tf_dataset = val_tf_dataset.batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    if args.model == 'densenet':
        logger.info('Using densenet')
        base_model = DenseNet121(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

    elif args.model == 'swin':
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
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,         # Number of epochs with no improvement after which training will be stopped
        verbose=1,
        restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity.
    )

    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=[AUC(), tfa.metrics.F1Score(num_classes=num_classes, average='macro')])
    
    if args.class_imbalance:
        history = model.fit(train_tf_dataset,
                            epochs=args.epochs,
                            validation_data=val_tf_dataset,
                            callbacks=[early_stopping],
                            class_weight=train_tf_dataset.get_class_weights()
                            )
    else: 
        history = model.fit(train_tf_dataset,
                            epochs=args.epochs,
                            validation_data=val_tf_dataset,
                            callbacks=[early_stopping]
                            )
    
    hist_df = pd.DataFrame(history.history)

    hist_df.to_csv(f'{log_dir}/{args.model}-{args.class_imbalance}.csv', index=False)
    logger.info(f"Saved model history to model_history/{args.model}-{args.class_imbalance}.csv")

    save_plot(history, 'auc', 'val_auc', 'AUC', f'{log_dir}/{args.model}-{args.class_imbalance}-auc-plot.png')
    save_plot(history, 'loss', 'val_loss', 'Loss', f'{log_dir}/{args.model}-{args.class_imbalance}-loss-plot.png')
    save_plot(history, 'f1_score', 'val_f1_score', 'F1 Score', f'{log_dir}/{args.model}-{args.class_imbalance}-f1-score-plot.png')

    explain_model(model, train_tf_dataset, output_file=f'{log_dir}/{args.model}-{args.class_imbalance}-explanation.png')
    logger.info(f"Saved model explanation to model_history/{args.model}-{args.class_imbalance}-explanation.png")
if __name__ == "__main__":
    model_choices = ['densenet','swin']

    parser = argparse.ArgumentParser(description="Arguments for training with pytorch")
    parser.add_argument("-m", "--model", choices=model_choices, help="Model to run", required=True)
    parser.add_argument("-im", "--class_imbalance", help="Handle class imbalance", required=False, default=False)
    parser.add_argument("-date", "--date", help="Start time", type=str, required=True)

    parser.add_argument("-epochs", "--epochs", help="Epochs", type=int, default=25)
    parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, default=32)

    
    args = parser.parse_args()
    args.class_imbalance = str_to_bool(args.class_imbalance)
    train(args)


