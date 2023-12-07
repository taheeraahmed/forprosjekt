import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score
import time
from utils.set_up import calculate_idun_time_left
import torchvision
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

class TrainerMultiClass:
    def __init__(self, model, model_output_folder, logger, optimizer, log_dir='runs', class_weights=None):
        self.model = model
        self.model_output_folder = model_output_folder
        self.logger = logger
        self.optimizer = optimizer
        self.classnames = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 
               'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 
               'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
        
        # moving model to device if cuda available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.logger.warning('GPU not available')

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        # for checkpointing
        self.best_val_f1 = 0.0

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
            logits = outputs if model_arg == 'densenet' else outputs.logits
            logits = logits.to(self.device)  # Ensure logits are on the correct device
            targets = labels.to(self.device)  # Ensure targets are on the correct device

            self.logger.info(f"Inputs device: {inputs.device}")
            self.logger.info(f"Labels device: {targets.device}")
            self.logger.info(f"Logits device: {logits.device}")

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

            # # Calculate per-class metrics
            # for cls_idx, cls_name in enumerate(self.classnames):
            #     cls_loss = self.criterion(logits[:, cls_idx], targets[:, cls_idx]).item()
            #     train_class_losses[cls_name] += cls_loss

            #     cls_correct_predictions = np.sum(outputs_binary[:, cls_idx] == targets_binary[:, cls_idx])
            #     train_class_correct[cls_name] += cls_correct_predictions
            #     train_class_total[cls_name] += targets_binary.shape[0]

            # Calculate and accumulate accuracy
            train_correct_predictions += np.sum(outputs_binary == targets_binary)
            train_total_predictions += targets_binary.size

            if i % 2 == 0:
                img_grid = torchvision.utils.make_grid(inputs)
                self.writer.add_image(f'Epoch {epoch}/four_xray_images', img_grid)

        # concatenate all outputs and targets
        train_outputs = np.vstack(train_outputs)
        train_targets = np.vstack(train_targets)

        # calculate average metrics for training
        avg_train_loss = train_loss / len(train_dataloader)
        train_f1 = f1_score(train_targets, train_outputs, average='macro')
        train_accuracy = np.mean(train_targets == train_outputs)
        
        # log training metrics
        self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        self.writer.add_scalar('F1/Train', train_f1, epoch)
        self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        # calculate AUC
        try:
            train_auc = roc_auc_score(train_targets, train_outputs, average='macro')
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

                # forward pass
                outputs = self.model(inputs)
                logits = outputs if model_arg == 'densenet' else outputs.logits
                logits = logits.to(self.device)  # Ensure logits are on the correct device
                targets = labels.to(self.device)  # Ensure targets are on the correct device

                #criterion
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
            val_auc = roc_auc_score(val_targets, val_outputs, average='macro')
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