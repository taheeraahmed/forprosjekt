import torch
import torch.nn as nn
import torch.optim as optim
from utils.plot_stuff import plot_metrics
from sklearn.metrics import accuracy_score, f1_score
import time
from utils.set_up import calculate_idun_time_left
import numpy as np
from tqdm import tqdm

class TrainingModuleBinaryClass:
    def __init__(self, model, args, logger, model_output_folder, output_folder, idun_time_done):
        self.model = model
        self.args = args
        self.logger = logger
        self.model_output_folder = model_output_folder
        self.idun_time_done = idun_time_done
        self.output_folder = output_folder

        self.best_val_accuracy = 0.0

        self.train_f1 = []
        self.train_accuracy = []
        self.train_losses = []

        self.val_f1 = []
        self.val_accuracy = []
        self.val_losses = []

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def _train_epoch(self, train_dataloader, epoch):
        self.model.train()
        train_loss, train_preds, train_targets = 0.0, [], []
        for inputs, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{self.args.num_epochs}"):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = outputs.squeeze(1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())

        train_loss /= len(train_dataloader)
        self.train_losses.append(train_loss)

        # Convert predictions to binary (if necessary)
        train_preds = np.array(train_preds)
        train_preds_binary = np.round(train_preds)  # Adjust this based on your use case

        # Calculate metrics
        self.train_f1.append(f1_score(train_targets, train_preds_binary, average='weighted'))
        self.train_accuracy.append(accuracy_score(train_targets, train_preds_binary))

    def _validate_epoch(self, val_dataloader, epoch):
        self.model.eval()

        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{self.args.num_epochs}"):
                outputs = self.model(inputs)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(labels.detach().cpu().numpy())

        val_loss /= len(val_dataloader)
        self.val_losses.append(val_loss)

        # Convert predictions to binary (if necessary)
        val_preds = np.array(val_preds)
        val_preds_binary = np.round(val_preds)  # Adjust this based on your use case

        # Calculate metrics
        self.val_f1.append(f1_score(val_targets, val_preds_binary, average='weighted'))
        self.val_accuracy.append(accuracy_score(val_targets, val_preds_binary))

        current_val_accuracy = self.val_accuracy[-1]
        if current_val_accuracy > self.best_val_accuracy:
            self.save_checkpoint(epoch, current_val_accuracy)
            self.best_val_f1 = current_val_accuracy

    def save_checkpoint(self, epoch, current_val_accuracy):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1
        }
        torch.save(checkpoint, f'{self.model_output_folder}/model_checkpoint_epoch_{epoch+1}.pt')
        self.logger.info(f'Checkpoint saved for epoch {epoch+1} with validation accuracy: {current_val_accuracy}')

    def train_model(self, train_dataloader, validation_dataloader, num_epochs):
        for epoch in range(num_epochs):
            epoch_start_time = time.time() 

            self.logger.info(f'Starting epoch {epoch+1}/{num_epochs}')

            self._train_epoch(train_dataloader, epoch)
            self._validate_epoch(validation_dataloader, epoch)

            self.logger.info(f'Epoch {epoch + 1}: , val_accuracy: {self.val_accuracy[-1]}, val_losses: {self.val_losses[-1]}')
            self.logger.info(f'Epoch {epoch + 1}: , train_accuracy: {self.train_accuracy[-1]}, train_losses: {self.train_losses[-1]}')
            self.scheduler.step()
            epoch_end_time = time.time()  # End time of the current epoch
            epoch_duration = epoch_end_time - epoch_start_time
            # Compare IDUN time with completion time and log
            calculate_idun_time_left(epoch, self.args.num_epochs, epoch_duration, self.idun_datetime_done, self.logger)

        plot_metrics(train_arr=self.train_losses, val_arr=self.val_losses, output_folder=self.output_folder, logger=self.logger, type='loss')
        plot_metrics(train_arr=self.train_f1, val_arr=self.val_f1, output_folder=self.output_folder, logger=self.logger, type='f1')
        plot_metrics(train_arr=self.train_accuracy, val_arr=self.val_accuracy, output_folder=self.output_folder, logger=self.logger, type='accuracy')