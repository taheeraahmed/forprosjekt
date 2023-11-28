from utils.set_up import set_up
from torch.utils.data import DataLoader
from datasets import ChestXrayDataset, get_binary_classification_df
from models import DenseNetBinaryClassifier
from torchvision import transforms
from tqdm import tqdm
import math
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
TODO: 
[x] Add test mode
[X] Add plots of train and val loss 
[X] f1 scores, accuracy, recall, precision
[X] Add plots of all

"""

def train(args):
    start_time = time.time()
    test = args.test
    output_folder = 'output/'+args.output_folder

    logger = set_up(output_folder=output_folder)
    logger.info(f'Output folder: {output_folder}')
    test_size = 0.2
    resize_size = (256, 256)
    lr = 0.001

    if test:
        logger.warning(f'In test mode')
        num_epochs = 2
        batch_size = 3
    else:  
        num_epochs= 30
        batch_size = 32

    logger.info(f'size: {resize_size}, test_size: {test_size}, batch_size: {batch_size}, num_epochs: {num_epochs}, lr: {lr}')
    data_path = '/cluster/home/taheeraa/datasets/chestxray-14'
    
    df = get_binary_classification_df(logger, data_path)
    transform = transforms.Compose([
            transforms.Resize(resize_size),  # Resize to 224x224
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing for pre-trained models
        ])
    train_df, val_df = train_test_split(df, test_size=test_size)  # Adjust the test_size as needed
    
    if test:
        test_df_size = 100
        val_df_size = math.floor(test_size*test_df_size)
        train_df = train_df.iloc[:test_df_size, :]
        val_df = val_df.iloc[:val_df_size, :]

    logger.info(f"Train df shape: {train_df.shape}")
    logger.info(f"Validation df shape: {val_df.shape}")

    train_dataset = ChestXrayDataset(df=train_df, transform=transform)
    val_dataset = ChestXrayDataset(df=val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DenseNetBinaryClassifier(logger=logger)
    model.log_params()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info('Started training')
    train_losses, val_losses = [], []
    train_f1, train_precision, train_recall, train_accuracy = [], [], [], []
    val_f1, val_precision, val_recall, val_accuracy = [], [], [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Convert predictions to binary (if necessary)
        train_preds = np.array(train_preds)
        train_preds_binary = np.round(train_preds)  # Adjust this based on your use case

        # Calculate metrics
        train_f1.append(f1_score(train_targets, train_preds_binary, average='weighted'))
        train_precision.append(precision_score(train_targets, train_preds_binary, average='weighted', zero_division=1))
        train_recall.append(recall_score(train_targets, train_preds_binary, average='weighted', zero_division=1))
        train_accuracy.append(accuracy_score(train_targets, train_preds_binary))

        # Validation phase
        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(labels.detach().cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Convert predictions to binary (if necessary)
        val_preds = np.array(val_preds)
        val_preds_binary = np.round(val_preds)  # Adjust this based on your use case

        # Calculate metrics
        val_f1.append(f1_score(val_targets, val_preds_binary, average='weighted'))
        val_precision.append(precision_score(val_targets, val_preds_binary, average='weighted', zero_division=1))
        val_recall.append(recall_score(val_targets, val_preds_binary, average='weighted', zero_division=1))
        val_accuracy.append(accuracy_score(val_targets, val_preds_binary))

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time} seconds")


    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_folder}/plot_train_val_loss.png')
    logger.info(f'Saved images to: {output_folder}/plot_train_val_loss.png')

    plt.figure(figsize=(10, 5))
    plt.plot(train_f1, label='Training F1')
    plt.plot(val_f1, label='Validation F1')
    plt.title('Training and Validation F1 Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig(f'{output_folder}/plot_F1.png')
    logger.info(f'Saved images to: {output_folder}/plot_F1.png')

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy, label='Training accuracy')
    plt.plot(val_accuracy, label='Validation accuracy')
    plt.title('Training and Validation Accuracy Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{output_folder}/plot_accuracy.png')
    logger.info(f'Saved images to: {output_folder}/plot_accuracy.png')


    plt.figure(figsize=(10, 5))
    plt.plot(train_precision, label='Training precision')
    plt.plot(val_precision, label='Validation precision')
    plt.title('Training and Validation Precision Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'{output_folder}/plot_precision.png')
    logger.info(f'Saved images to: {output_folder}/plot_precision.png')

    plt.figure(figsize=(10, 5))
    plt.plot(train_recall, label='Training recall')
    plt.plot(val_recall, label='Validation recall')
    plt.title('Training and Validation Recall Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig(f'{output_folder}/plot_recall.png')
    logger.info(f'Saved images to: {output_folder}/plot_recall.png')

def str_to_bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {value} to boolean.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training with pytorch")
    parser.add_argument("-t", "--test", help="Test mode?", default=True, required=True)
    parser.add_argument("-of", "--output_folder", help="Name of folder output files will be added", required=True)
    args = parser.parse_args()
    args.test = str_to_bool(args.test)
    train(args)