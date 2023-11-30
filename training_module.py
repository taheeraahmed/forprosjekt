import torch
import torch.nn as nn
import torch.optim as optim
from utils.plot_stuff import plot_metrics
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import time
from utils.set_up import calculate_idun_time_left

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TrainingModuleMultiClass:
    def __init__(self, model, logger, learning_rate, step_size=5, gamma=0.1, log_dir='runs'):
        # Initialize the model
        self.model = model
        self.logger = logger
        self.classnames = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
               'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
               'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters())
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir)

    def train(self, train_dataloader, val_dataloader, num_epochs):
        for epoch in range(num_epochs):
            self._train_epoch(train_dataloader, epoch)
            self._validate_epoch(val_dataloader, epoch)
            self.scheduler.step()

        self.writer.close()

    def _train_epoch(self, train_dataloader, epoch):
        self.model.train()

        # Variables to store metrics for training
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0

        # Initialize per-class metrics storage for training
        train_class_losses = {classname: 0.0 for classname in self.classnames}
        train_class_correct = {classname: 0 for classname in self.classnames}
        train_class_total = {classname: 0 for classname in self.classnames}

        train_loop = tqdm(train_dataloader, leave=True)
        for i, batch in enumerate(train_loop):
            outputs = self.model(batch["img"])
            targets = batch["lab"][:, :, None].squeeze(-1)
            loss = self.criterion(outputs, targets)
            
            # Perform backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Accumulate training loss
            train_loss += loss.item()

            # Convert outputs and targets to binary format for each class
            outputs_binary = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            targets_binary = targets.cpu().numpy()

            # Calculate per-class metrics
            for cls_idx, cls_name in enumerate(self.classnames):
                cls_loss = self.criterion(outputs[:, cls_idx], targets[:, cls_idx]).item()
                train_class_losses[cls_name] += cls_loss

                cls_correct_predictions = np.sum(outputs_binary[:, cls_idx] == targets_binary[:, cls_idx])
                train_class_correct[cls_name] += cls_correct_predictions
                train_class_total[cls_name] += targets_binary.shape[0]

            # Calculate and accumulate accuracy and F1 score
            train_correct_predictions += np.sum(outputs_binary == targets_binary)
            train_total_predictions += targets_binary.size

            if i % 2 == 0:
                img_grid = torchvision.utils.make_grid(batch["img"])
                self.writer.add_image('four_xray_images', img_grid)

        # Calculate average metrics for training
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_accuracy = train_correct_predictions / train_total_predictions
        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', avg_train_accuracy, epoch)

        # Calculate and log per-class metrics for training
        for cls_name in self.classnames:
            avg_cls_loss = train_class_losses[cls_name] / len(train_dataloader)
            cls_accuracy = train_class_correct[cls_name] / train_class_total[cls_name]
            self.writer.add_scalar(f'Train/Loss/{cls_name}', avg_cls_loss, epoch)
            self.writer.add_scalar(f'Train/Accuracy/{cls_name}', cls_accuracy, epoch)
            
    def _validate_epoch(self, validation_dataloader, epoch):
        # Validation loop
        self.model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        # Initialize per-class metrics storage for validation
        val_class_losses = {classname: 0.0 for classname in self.classnames}
        val_class_correct = {classname: 0 for classname in self.classnames}
        val_class_total = {classname: 0 for classname in self.classnames}

        with torch.no_grad():
            val_loop = tqdm(validation_dataloader, leave=True)
            for i, batch in enumerate(val_loop):
                outputs = self.model(batch["img"])
                targets = batch["lab"][:, :, None].squeeze(-1)
                loss = self.criterion(outputs, targets)

                # Accumulate validation loss
                val_loss += loss.item()

                outputs_binary = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                targets_binary = targets.cpu().numpy()

                # Calculate per-class metrics
                for cls_idx, cls_name in enumerate(self.classnames):
                    cls_loss = self.criterion(outputs[:, cls_idx], targets[:, cls_idx]).item()
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
        self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', avg_val_accuracy, epoch)

        for cls_name in self.classnames:
            avg_cls_loss = val_class_losses[cls_name] / len(validation_dataloader)
            cls_accuracy = val_class_correct[cls_name] / val_class_total[cls_name]
            self.writer.add_scalar(f'Validation/Loss/{cls_name}', avg_cls_loss, epoch)
            self.writer.add_scalar(f'Validation/Accuracy/{cls_name}', cls_accuracy, epoch)

class TrainingModuleBinaryClass:
    def __init__(self, model, train_dataloader, validation_dataloader, args, logger, model_output_folder, output_folder, idun_time_done):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
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

    def train_epoch(self, epoch):
        self.model.train()
        train_loss, train_preds, train_targets = 0.0, [], []
        for inputs, labels in tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}/{self.args.num_epochs}"):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = outputs.squeeze(1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())

        train_loss /= len(self.train_dataloader)
        self.train_losses.append(train_loss)

        # Convert predictions to binary (if necessary)
        train_preds = np.array(train_preds)
        train_preds_binary = np.round(train_preds)  # Adjust this based on your use case

        # Calculate metrics
        self.train_f1.append(f1_score(train_targets, train_preds_binary, average='weighted'))
        self.train_accuracy.append(accuracy_score(train_targets, train_preds_binary))

    def validate_epoch(self,epoch):
        self.model.eval()

        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in tqdm(self.validation_dataloader, desc=f"Validation Epoch {epoch + 1}/{self.args.num_epochs}"):
                outputs = self.model(inputs)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(outputs.detach().cpu().numpy())
                val_targets.extend(labels.detach().cpu().numpy())

        val_loss /= len(self.validation_dataloader)
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

    def save_checkpoint(self, epoch, current_val_accuracy):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy
        }
        torch.save(checkpoint, f'{self.model_output_folder}/model_checkpoint_epoch_{epoch+1}.pt')
        self.logger.info(f'Checkpoint saved for epoch {epoch+1} with validation accuracy: {current_val_accuracy}')

    def train_model(self):
        for epoch in range(self.args.num_epochs):
            epoch_start_time = time.time() 
            self.logger.info(f'Starting epoch {epoch+1}/{self.args.num_epochs}')
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
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