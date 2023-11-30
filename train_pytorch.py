from utils.set_up import set_up, calculate_idun_time_left
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from datasets import ChestXrayDataset, get_binary_classification_df, ModifiedNIH_Dataset
from utils.create_dir import create_directory_if_not_exists
from utils.plot_stuff import plot_metrics, plot_pred
import tqdm
from models import DenseNetBinaryClassifier
import torchvision
from torch.utils.data import Subset
from tqdm import tqdm
import math
import argparse
import time
import numpy as np
import csv
import sys

import torch
import torchxrayvision as xrv
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def list_directories(path):
    return [os.path.join(d, f'{path}/{d}/images') for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

CLASSNAMES = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
               'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
               'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
"""
TODO: 
    X Add validation loop as well
    X Fix loss in Tensorboard
    X Fix not values not being logged in tensorboard aaah
    X Delete all tensorboard files
    Start a real run B)
"""
def train(args):
    
    start_time = time.time()

    # Setting argument variables
    test = args.test
    idun_time = args.idun_time
    output_folder = 'output/'+args.output_folder
    model_arg = args.model

    # Creating directory for output, creating logger and calculating IDUN job done datetime
    create_directory_if_not_exists(output_folder+'/models')
    model_output_folder = output_folder+'/models'
    logger, idun_datetime_done = set_up(output_folder=output_folder, idun_time=idun_time, start_time=start_time)
    logger.info(f'Output folder: {output_folder}')

    # Setting params
    test_size = 0.2
    lr = 0.001
    num_epochs= 15
    batch_size = 32

    data_path = '/cluster/home/taheeraa/datasets/chestxray-14'

    if test:
        logger.warning(f'In test mode')
        num_epochs = 2
        batch_size = 3

    logger.info(f'test_size: {test_size}, batch_size: {batch_size}, num_epochs: {num_epochs}, lr: {lr}')

    if model_arg == 'densenet-pretrained-xray-multi-class':
        data_transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)])
        nih_img_dirs = list_directories(data_path)
        logger.info(f'{nih_img_dirs}')
        dataset = ModifiedNIH_Dataset(imgpaths=nih_img_dirs, transform=data_transforms)
        if test: 
            logger.warning('Using 1% of dataset')

            subset_size = int(len(dataset) * 0.01)
            indices = torch.randperm(len(dataset)).tolist()
            test_subset_indices = indices[:subset_size]
            test_subset_dataset = Subset(dataset, test_subset_indices)

            # Split the test subset into test and validation sets
            test_size = int(0.8 * len(test_subset_dataset))
            validation_size = len(test_subset_dataset) - test_size
            test_dataset, validation_dataset = random_split(test_subset_dataset, [test_size, validation_size])

            # Create dataloaders for test and validation sets
            train_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        else:
            # Split the dataset into training and validation sets
            train_size = int(0.8 * len(dataset))
            validation_size = len(dataset) - train_size
            train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

            # Create dataloaders for training and validation sets
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        model = xrv.models.get_model(weights="densenet121-res224-nih")
        model.op_threshs = None # prevent pre-trained model calibration
        model.classifier = torch.nn.Linear(1024,14) # reinitialize classifier
        
        optimizer = torch.optim.Adam(model.classifier.parameters()) # only train classifier
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        writer = SummaryWriter('runs/xray_experiment_4')

        for epoch in range(num_epochs):
            epoch_start_time = time.time() 
            model.train()

            # Variables to store metrics for training
            train_loss = 0.0
            train_correct_predictions = 0
            train_total_predictions = 0

            # Initialize per-class metrics storage for training
            train_class_losses = {classname: 0.0 for classname in CLASSNAMES}
            train_class_correct = {classname: 0 for classname in CLASSNAMES}
            train_class_total = {classname: 0 for classname in CLASSNAMES}

            train_loop = tqdm(train_dataloader, leave=True)
            for i, batch in enumerate(train_loop):
                outputs = model(batch["img"])
                targets = batch["lab"][:, :, None].squeeze(-1)
                loss = criterion(outputs, targets)
                
                # Perform backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Accumulate training loss
                train_loss += loss.item()

                # Convert outputs and targets to binary format for each class
                outputs_binary = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                targets_binary = targets.cpu().numpy()

                # Calculate per-class metrics
                for cls_idx, cls_name in enumerate(CLASSNAMES):
                    cls_loss = criterion(outputs[:, cls_idx], targets[:, cls_idx]).item()
                    train_class_losses[cls_name] += cls_loss

                    cls_correct_predictions = np.sum(outputs_binary[:, cls_idx] == targets_binary[:, cls_idx])
                    train_class_correct[cls_name] += cls_correct_predictions
                    train_class_total[cls_name] += targets_binary.shape[0]

                # Calculate and accumulate accuracy and F1 score
                train_correct_predictions += np.sum(outputs_binary == targets_binary)
                train_total_predictions += targets_binary.size

                if i % 2 == 0:
                    img_grid = torchvision.utils.make_grid(batch["img"])
                    writer.add_image('four_xray_images', img_grid)

            # Calculate average metrics for training
            avg_train_loss = train_loss / len(train_dataloader)
            avg_train_accuracy = train_correct_predictions / train_total_predictions

            # Calculate and log per-class metrics for training
            for cls_name in CLASSNAMES:
                avg_cls_loss = train_class_losses[cls_name] / len(train_dataloader)
                cls_accuracy = train_class_correct[cls_name] / train_class_total[cls_name]
                writer.add_scalar(f'Train/Loss/{cls_name}', avg_cls_loss, epoch)
                writer.add_scalar(f'Train/Accuracy/{cls_name}', cls_accuracy, epoch)

            # Validation loop
            model.eval()
            val_loss = 0.0
            val_correct_predictions = 0
            val_total_predictions = 0

            # Initialize per-class metrics storage for validation
            val_class_losses = {classname: 0.0 for classname in CLASSNAMES}
            val_class_correct = {classname: 0 for classname in CLASSNAMES}
            val_class_total = {classname: 0 for classname in CLASSNAMES}

            with torch.no_grad():
                val_loop = tqdm(validation_dataloader, leave=True)
                for i, batch in enumerate(val_loop):
                    outputs = model(batch["img"])
                    targets = batch["lab"][:, :, None].squeeze(-1)
                    loss = criterion(outputs, targets)

                    # Accumulate validation loss
                    val_loss += loss.item()

                    outputs_binary = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                    targets_binary = targets.cpu().numpy()

                    # Calculate per-class metrics
                    for cls_idx, cls_name in enumerate(CLASSNAMES):
                        cls_loss = criterion(outputs[:, cls_idx], targets[:, cls_idx]).item()
                        val_class_losses[cls_name] += cls_loss

                        cls_correct_predictions = np.sum(outputs_binary[:, cls_idx] == targets_binary[:, cls_idx])
                        val_class_correct[cls_name] += cls_correct_predictions
                        val_class_total[cls_name] += targets_binary.shape[0]

                    # Calculate and accumulate accuracy and F1 score
                    val_correct_predictions += np.sum(outputs_binary == targets_binary)
                    val_total_predictions += targets_binary.size

            # Calculate average metrics for validation
            avg_val_loss = val_loss / len(validation_dataloader)
            avg_val_accuracy = val_correct_predictions / val_total_predictions

            for cls_name in CLASSNAMES:
                avg_cls_loss = val_class_losses[cls_name] / len(validation_dataloader)
                cls_accuracy = val_class_correct[cls_name] / val_class_total[cls_name]
                writer.add_scalar(f'Validation/Loss/{cls_name}', avg_cls_loss, epoch)
                writer.add_scalar(f'Validation/Accuracy/{cls_name}', cls_accuracy, epoch)

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Accuracy/Train', avg_train_accuracy, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/Validation', avg_val_accuracy, epoch)

            # Log elapsed time for epoch
            epoch_end_time = time.time() 
            epoch_duration = epoch_end_time - epoch_start_time

            calculate_idun_time_left(epoch, num_epochs, epoch_duration, idun_datetime_done, logger)
            logger.info(f'Epoch {epoch+1} - Train loss: {avg_train_loss}, Train accuracy: {avg_train_accuracy}, Val loss: {avg_val_loss}, Val accuracy: {avg_val_accuracy}')

        writer.close()
    elif model_arg == 'densenet-pretrained-imagenet-binary-class':
        df = get_binary_classification_df(logger, data_path)
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
        resize_size = (256, 256)
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(resize_size), 
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            ])
        train_dataset = ChestXrayDataset(df=train_df, transform=transform)
        val_dataset = ChestXrayDataset(df=val_df, transform=transform)
        model = DenseNetBinaryClassifier(logger=logger)
        model.log_params()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info('Started training HLO')
        train_losses, val_losses = [], []
        train_f1, train_precision, train_recall, train_accuracy = [], [], [], []
        val_f1, val_precision, val_recall, val_accuracy = [], [], [], []

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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

                # Writing to csv
                train_metrics = [epoch + 1, train_accuracy[-1], train_f1[-1], train_recall[-1], train_precision[-1], train_losses[-1]]
                val_metrics = [epoch + 1, val_accuracy[-1], val_f1[-1], val_recall[-1], val_precision[-1], val_losses[-1]]
                train_writer.writerow(train_metrics)
                val_writer.writerow(val_metrics)
                train_file.flush()
                val_file.flush()

                logger.info(f'Epoch {epoch + 1}: , {val_accuracy[-1]}, {val_f1[-1]}, {val_recall[-1]}, {val_precision[-1]}, {val_losses[-1]}')
                logger.info(f'Epoch {epoch + 1}: , {train_accuracy[-1]}, {train_f1[-1]}, {train_recall[-1]}, {train_precision[-1]}, {train_losses[-1]}')

                # Elapsed time
                epoch_end_time = time.time()  # End time of the current epoch
                epoch_duration = epoch_end_time - epoch_start_time

                # Compare IDUN time with completion time and log
                calculate_idun_time_left(epoch, num_epochs, epoch_duration, idun_datetime_done, logger)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time: {elapsed_time} seconds")

        plot_metrics(train_arr=train_losses, val_arr=val_losses, output_folder=output_folder, logger=logger, type='loss')
        plot_metrics(train_arr=train_f1, val_arr=val_f1, output_folder=output_folder, logger=logger, type='f1')
        plot_metrics(train_arr=train_accuracy, val_arr=val_accuracy, output_folder=output_folder, logger=logger, type='accuracy')
        plot_metrics(train_arr=train_precision, val_arr=val_precision, output_folder=output_folder, logger=logger, type='precision')
        plot_metrics(train_arr=train_recall, val_arr=val_recall, output_folder=output_folder, logger=logger, type='recall')
        
        inputs, labels = next(iter(val_loader))     # Get a batch of training data
        outputs = model(inputs)                     # Make predictions     
        preds = torch.sigmoid(outputs) > 0.5        # Assuming binary classification with sigmoid activation

        plot_pred(inputs, labels, preds, output_folder, logger)
    else: 
        logger.error('Invalid model argument')
        sys.exit(1)
    

def str_to_bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {value} to boolean.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training with pytorch")
    parser.add_argument("-of", "--output_folder", help="Name of folder output files will be added", required=False, default='./output/')
    parser.add_argument("-it", "--idun_time", help="The duration of job set on IDUN", default=None, required=False)
    parser.add_argument("-t", "--test", help="Test mode?", required=False, default=True)
    parser.add_argument("-m", "--model", choices=["densenet-pretrained-xray-multi-class", "densenet-pretrained-imagenet-binary-class"], help="Model to run", required=True)
    
    args = parser.parse_args()
    args.test = str_to_bool(args.test)
    train(args)