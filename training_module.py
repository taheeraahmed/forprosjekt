import torch
import torch.nn as nn
import torch.optim as optim
from utils.plot_stuff import plot_metrics
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
from utils.set_up import calculate_idun_time_left
import torchvision
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

class TrainingModuleMultiClass:
    def __init__(self, model, model_output_folder, logger, step_size=5, gamma=0.1, log_dir='runs', lr=0.001):
        self.model = model
        self.logger = logger
        self.classnames = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
               'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
               'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.optimizer = torch.optim.Adam(self.model.classifier.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.model_output_folder = model_output_folder

        # for checkpointing
        self.best_val_f1 = 0.0

        # moving model to device if cuda available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.logger.warning('GPU not available')

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)

    def train(self, train_dataloader, validation_dataloader, num_epochs, idun_datetime_done, model_arg):
        for epoch in range(num_epochs):
            epoch_start_time = time.time() 
            self.logger.info(f'Started epoch {epoch+1}')
            
            self._train_epoch(train_dataloader, epoch, model_arg)
            self._validate_epoch(validation_dataloader, epoch, model_arg)
            self.scheduler.step()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            calculate_idun_time_left(epoch, num_epochs, epoch_duration, idun_datetime_done, self.logger)

        self.writer.close()

    def _save_checkpoint(self, epoch, current_val_accuracy):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1
        }
        torch.save(checkpoint, f'{self.model_output_folder}/model_checkpoint_epoch_{epoch+1}.pt')
        self.logger.info(f'Checkpoint saved for epoch {epoch+1} with f1 score: {current_val_accuracy}')

    def _train_epoch(self, train_dataloader, epoch, model_arg):
        self.model.train()

        # vars to store metrics for training
        train_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0
        train_outputs = []
        train_targets = []

        # init per-class metrics storage for training
        train_class_losses = {classname: 0.0 for classname in self.classnames}
        train_class_correct = {classname: 0 for classname in self.classnames}
        train_class_total = {classname: 0 for classname in self.classnames}

        train_loop = tqdm(train_dataloader, leave=True)
        for i, batch in enumerate(train_loop):
            inputs, labels = batch["img"].to(self.device), batch["lab"].to(self.device)
            self.optimizer.zero_grad()
            
            # forward pass
            outputs = self.model(inputs)
            if model_arg == 'densenet-pretrained-xray-multi-class' or 'densenet-pretrained-xray-multi-class-imbalance':
                logits = outputs = self.model(inputs)
            else:
                logits = outputs.logits
            targets = labels

            # compute loss
            loss = self.criterion(logits, targets)
            
            # backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # accumulate training loss
            train_loss += loss.item()

            # convert outputs (logits) and targets to binary format for each class
            outputs_binary = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            targets_binary = targets.cpu().numpy()

            # appending for roc-auc
            train_outputs.append(outputs_binary)
            train_targets.append(targets_binary)

            # Calculate per-class metrics
            for cls_idx, cls_name in enumerate(self.classnames):
                cls_loss = self.criterion(logits[:, cls_idx], targets[:, cls_idx]).item()
                train_class_losses[cls_name] += cls_loss

                cls_correct_predictions = np.sum(outputs_binary[:, cls_idx] == targets_binary[:, cls_idx])
                train_class_correct[cls_name] += cls_correct_predictions
                train_class_total[cls_name] += targets_binary.shape[0]

            # Calculate and accumulate accuracy
            train_correct_predictions += np.sum(outputs_binary == targets_binary)
            train_total_predictions += targets_binary.size

            if i % 2 == 0:
                img_grid = torchvision.utils.make_grid(inputs)
                self.writer.add_image(f'Epoch {epoch}/four_xray_images', img_grid)

        # calculate average metrics for training
        avg_train_loss = train_loss / len(train_dataloader)
        train_f1 = f1_score(targets_binary, outputs_binary, average='weighted')
        train_accuracy = train_correct_predictions / train_total_predictions
        

        # log training metrics
        self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        self.writer.add_scalar('F1/Train', train_f1, epoch)
        self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        # calculate AUC
        try:
            train_outputs = np.vstack(train_outputs)
            train_targets = np.vstack(train_targets)
            train_auc = roc_auc_score(train_targets, train_outputs, average='macro')  # or 'micro'
            self.writer.add_scalar('AUC/Train', train_auc, epoch)
            self.logger.info(f'[Train] Epoch {epoch+1} - loss: {avg_train_loss}, F1: {train_f1}, auc: {train_auc}, accuracy: {train_accuracy}')
        except ValueError as e:
            self.logger.warning(f'Unable to calculate train AUC for epoch {epoch+1}: {e}')
            self.logger.info(f'[Train] Epoch {epoch+1} - loss: {avg_train_loss}, F1: {train_f1}, accuracy: {train_accuracy}')

        # calculate and log per-class metrics for training
        for cls_name in self.classnames:
            avg_cls_loss = train_class_losses[cls_name] / len(train_dataloader)
            cls_accuracy = train_class_correct[cls_name] / train_class_total[cls_name]
            self.writer.add_scalar(f'Train/Loss/{cls_name}', avg_cls_loss, epoch)
            self.writer.add_scalar(f'Train/Accuracy/{cls_name}', cls_accuracy, epoch)
            
    def _validate_epoch(self, validation_dataloader, epoch, model_arg):
        self.model.eval()
        
        # vars to store metrics
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        val_outputs = []
        val_targets = []

        # init per-class metrics storage for validation
        val_class_losses = {classname: 0.0 for classname in self.classnames}
        val_class_correct = {classname: 0 for classname in self.classnames}
        val_class_total = {classname: 0 for classname in self.classnames}

        with torch.no_grad():
            val_loop = tqdm(validation_dataloader, leave=True)
            for i, batch in enumerate(val_loop):
                inputs, labels = batch["img"].to(self.device), batch["lab"].to(self.device)
                outputs = self.model(inputs)
                # Need to make it fit for the transformer and densenet--- Hacky :) 
                if model_arg == 'densenet-pretrained-xray-multi-class':
                    logits = outputs = self.model(inputs)
                else:
                    logits = outputs.logits
                targets = labels
                loss = self.criterion(logits, targets)

                # accumulate validation loss
                val_loss += loss.item()

                outputs_binary = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                targets_binary = targets.cpu().numpy()

                # calculate per-class metrics
                for cls_idx, cls_name in enumerate(self.classnames):
                    cls_loss = self.criterion(logits[:, cls_idx], targets[:, cls_idx]).item()
                    val_class_losses[cls_name] += cls_loss

                    cls_correct_predictions = np.sum(outputs_binary[:, cls_idx] == targets_binary[:, cls_idx])
                    val_class_correct[cls_name] += cls_correct_predictions
                    val_class_total[cls_name] += targets_binary.shape[0]

                # calculate and accumulate accuracy, auc and F1 score
                val_correct_predictions += np.sum(outputs_binary == targets_binary)
                val_total_predictions += targets_binary.size
                val_outputs.append(outputs_binary)
                val_targets.append(targets_binary)
        
        
        # calculate average metrics for validation
        avg_val_loss = val_loss / len(validation_dataloader)
        val_f1 = f1_score(targets_binary, outputs_binary, average='macro')  
        val_accuracy = val_correct_predictions / val_total_predictions

        # write to tensorboard
        self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        self.writer.add_scalar('F1/Validation', val_f1, epoch)
        self.writer.add_scalar('Accuracy/Train', val_accuracy, epoch)

        # log and check if possible to calculate AUC
        try:
            val_outputs = np.vstack(val_outputs)
            val_targets = np.vstack(val_targets)
            val_auc = roc_auc_score(val_targets, val_outputs, average='macro')  # or 'micro'
            self.writer.add_scalar('AUC/Validation', val_auc, epoch)
            self.logger.info(f'[Validation] Epoch {epoch+1} - loss: {avg_val_loss}, F1: {val_f1}, auc: {val_auc}, accuracy: {val_accuracy}')
        except ValueError as e:
            self.logger.warning(f'Unable to calculate validation AUC for epoch {epoch+1}: {e}')
            self.logger.info(f'[Validation] Epoch {epoch+1} - loss: {avg_val_loss}, F1: {val_f1}, accuracy: {val_accuracy}')
        
        # checkpointing
        current_val_f1 = val_f1
        if current_val_f1 > self.best_val_f1:
            self.best_val_f1 = current_val_f1
            self._save_checkpoint(epoch, current_val_f1)

        for cls_name in self.classnames:
            avg_cls_loss = val_class_losses[cls_name] / len(validation_dataloader)
            cls_accuracy = val_class_correct[cls_name] / val_class_total[cls_name]
            self.writer.add_scalar(f'Validation/Loss/{cls_name}', avg_cls_loss, epoch)
            self.writer.add_scalar(f'Validation/Accuracy/{cls_name}', cls_accuracy, epoch)

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
        self.train_f1.append(f1_score(train_targets, train_preds_binary, average='macro'))
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