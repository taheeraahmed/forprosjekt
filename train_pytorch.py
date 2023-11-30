from utils.set_up import set_up, calculate_idun_time_left
from datetime import datetime, timedelta
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
            logger.warning('Using 10% of dataset')
            subset_size = int(len(dataset) * 0.05) 
            indices = torch.randperm(len(dataset)).tolist()
            subset_indices = indices[:subset_size]
            subset_dataset = Subset(dataset, subset_indices)
            dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=batch_size)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        model = xrv.models.get_model(weights="densenet121-res224-nih")
        model.op_threshs = None # prevent pre-trained model calibration
        model.classifier = torch.nn.Linear(1024,14) # reinitialize classifier
        
        optimizer = torch.optim.Adam(model.classifier.parameters()) # only train classifier
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        writer = SummaryWriter('runs/xray_experiment_3')
        num_classes = 14
        for epoch in range(num_epochs):
            epoch_start_time = time.time() 
            model.train()
            loop = tqdm(dataloader, leave=True)
            for i, batch in enumerate(loop):
                outputs = model(batch["img"])
                targets = batch["lab"][:, :, None].squeeze(-1)
                loss = criterion(outputs, targets)
                
                # Perform backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Convert outputs and targets to binary format for each class
                outputs_binary = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                targets_binary = targets.cpu().numpy()

                # Calculate and log metrics
                correct_predictions = np.sum(outputs_binary == targets_binary)
                total_predictions = targets_binary.size
                combined_accuracy = correct_predictions / total_predictions

                for cls in range(num_classes):
                    cls_name = CLASSNAMES[cls]

                    cls_precision = precision_score(targets_binary[:, cls], outputs_binary[:, cls], zero_division=0)
                    cls_recall = recall_score(targets_binary[:, cls], outputs_binary[:, cls], zero_division=0)
                    cls_f1 = f1_score(targets_binary[:, cls], outputs_binary[:, cls], zero_division=0)
                    cls_accuracy = accuracy_score(targets_binary[:, cls], outputs_binary[:, cls])

                    writer.add_scalar(f'{cls_name}/Precision', cls_precision, epoch)
                    writer.add_scalar(f'{cls_name}/Recall', cls_recall, epoch)
                    writer.add_scalar(f'{cls_name}/F1', cls_f1, epoch)
                    writer.add_scalar(f'{cls_name}/Accuracy', cls_accuracy, epoch)

                # Update tqdm loop and log to TensorBoard
                loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
                loop.set_postfix(loss=loss.item(), acc=combined_accuracy)

                writer.add_scalar('Loss/Training', loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Accuracy/Train', combined_accuracy, epoch * len(dataloader) + i)
                
                
                # Make a grid from batch images
                if i % 2 == 0:
                    img_grid = torchvision.utils.make_grid(batch["img"])
                    writer.add_image('four_xray_images', img_grid)
                
            # Elapsed time
            logger.info(f'Epoch {epoch+1} - Loss {loss.item()}, accuracy {combined_accuracy[-1]}')
            epoch_end_time = time.time()  # End time of the current epoch
            epoch_duration = epoch_end_time - epoch_start_time

            calculate_idun_time_left(epoch, num_epochs, epoch_duration, idun_datetime_done, logger)

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