from utils.set_up import set_up
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from datasets import ChestXrayDataset, get_binary_classification_df
from utils.create_dir import create_directory_if_not_exists
from models import DenseNetBinaryClassifier
from torchvision import transforms
from tqdm import tqdm
import math
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and then add the mean
    return tensor

def train(args):
    start_time = time.time()
    test = args.test
    output_folder = 'output/'+args.output_folder
    create_directory_if_not_exists(output_folder+'/models')
    model_output_folder = output_folder+'/models'

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
        num_epochs= 15
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
    else: 
        train_df = train_df.sample(frac=0.25)
        val_df = val_df.sample(frac=0.25)

    logger.info(f"Train df shape: {train_df.shape}")
    logger.info(f"Validation df shape: {val_df.shape}")

    train_dataset = ChestXrayDataset(df=train_df, transform=transform)
    val_dataset = ChestXrayDataset(df=val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DenseNetBinaryClassifier(logger=logger)
    model.log_params()
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    logger.info('Started training')
    train_losses, val_losses = [], []
    train_f1, train_precision, train_recall, train_accuracy = [], [], [], []
    val_f1, val_precision, val_recall, val_accuracy = [], [], [], []

    best_val_accuracy = 0.0  # Initialize the best validation accuracy
    header = ['epoch','accuracy', 'f1', 'recall', 'precision']
    with open(f'{output_folder}/train_metrics.csv', mode='w', newline='') as train_file, \
        open(f'{output_folder}/val_metrics.csv', mode='w', newline='') as val_file:

        # Create CSV writers
        train_writer = csv.writer(train_file)
        val_writer = csv.writer(val_file)

        # Write the header
        train_writer.writerow(header)
        val_writer.writerow(header)

        for epoch in range(num_epochs):
            epoch_start_time = time.time() 
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

            # Update learning rate
            scheduler.step()

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

            # Check if the current validation accuracy is the best
            current_val_accuracy = val_accuracy[-1]  # Get the latest validation accuracy
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy
                }
                torch.save(checkpoint, f'{model_output_folder}/model_checkpoint_epoch_{epoch+1}.pt')
                logger.info(f'Checkpoint saved for epoch {epoch+1} with validation accuracy: {current_val_accuracy}')

            train_metrics = [epoch + 1, train_accuracy[-1], train_f1[-1], train_recall[-1], train_precision[-1], train_losses[-1]]
            val_metrics = [epoch + 1, val_accuracy[-1], val_f1[-1], val_recall[-1], val_precision[-1], val_losses[-1]]
            train_writer.writerow(train_metrics)
            val_writer.writerow(val_metrics)

            train_file.flush()
            val_file.flush()

            epoch_end_time = time.time()  # End time of the current epoch
            epoch_duration = epoch_end_time - epoch_start_time
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining_time = epoch_duration * remaining_epochs
            # Calculate the estimated completion time
            estimated_completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)


            logger.info(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")
            logger.info(f"Estimated time remaining: {estimated_remaining_time:.2f} seconds")
            logger.info(f"Estimated completion time: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}")


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

    # Get a batch of training data
    inputs, labels = next(iter(val_loader))

    # Make predictions
    outputs = model(inputs)
    preds = torch.sigmoid(outputs) > 0.5  # Assuming binary classification with sigmoid activation

    # Plot the images and labels
    num_images = len(inputs)
    cols = int(np.sqrt(num_images))
    rows = cols if cols**2 == num_images else cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust the space between images

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            input = inputs[i]
            denormalized_input = denormalize(input.clone(), mean, std)
            img = denormalized_input.numpy().transpose((1, 2, 0))
            plt.imshow(img, cmap='gray')
            actual_label = 'Positive' if labels[i].item() == 1 else 'Negative'
            predicted_label = 'Positive' if preds[i].item() == 1 else 'Negative'
            ax.set_title(f'Actual: {actual_label}\nPredicted: {predicted_label}', fontsize=10, backgroundcolor='white')
            ax.axis('off')  # Hide the axis
        else:
            ax.axis('off')  # Hide axis if no image

    plt.tight_layout()
    plt.savefig(f'{output_folder}/img_chest_pred.png')
    logger.info(f'Saved images to: {output_folder}/img_chest_pred.png')
    logger.info('Done training')

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