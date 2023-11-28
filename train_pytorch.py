from utils.set_up import set_up
from torch.utils.data import DataLoader
from datasets import ChestXrayDataset, get_binary_classification_df
from models import DenseNetBinaryClassifier
from torchvision import transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_accuracy(outputs, labels):
    # Convert outputs to predicted labels (0 or 1)
    predicted = outputs.round()
    correct = (predicted == labels).float()  # convert to float for division
    accuracy = correct.sum() / len(correct)
    return accuracy

def train(test = True):
    logger = set_up()
    logger.info(f'Test:{test}')
    test_size = 0.2
    if test:
        resize_size = (256, 256)
        num_epochs = 1
        batch_size = 3
    else:  
        resize_size = (32, 32)
        num_epochs= 100
        batch_size = 32

    logger.info(f'size: {resize_size}, test_size: {test_size}, batch_size: {batch_size}, num_epochs: {num_epochs}')
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
        train_df = train_df[:100]
        val_df = val_df[:100*test_size]

    train_dataset = ChestXrayDataset(df=train_df, transform=transform)
    val_dataset = ChestXrayDataset(df=val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DenseNetBinaryClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info('Started training')

    for epoch in range(num_epochs):
        #Training phase
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        for inputs, labels in train_progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update the progress bar with the loss information
            train_progress_bar.set_postfix(loss=loss.item())
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Calculate and log training metrics
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  # No need to track gradients during validation
            val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")
            for inputs, labels in val_progress_bar:
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                val_progress_bar.set_postfix(loss=loss.item())

                val_loss += loss.item()
        
        val_loss /= len(val_loader)
    

if __name__ == "__main__":
    train()