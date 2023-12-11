import torch.nn as nn
from torchvision import models
import torch
from src.dataloader import BinaryClassificationDataLoader
from trainers.trainer_multi_class import TrainerMultiClass
from trainers.trainer_binary_class import TrainingModuleBinaryClass
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchxrayvision as xrv
from datasets.multi_class import ChestXray14MultiClassDataset
from trainers.trainer_multi_class import TrainerMultiClass
from utils.handle_class_imbalance import handle_class_imbalance_df, get_class_weights

def densenet(logger, args, idun_datetime_done, data_path):
    # TODO: Handle class imbalance for multiclass only, do i need support for binary tho? Doubts :)
    if args.task == 'multi-class':
        logger.info('Multi classification')
        shuffle = True  
        num_workers = 4  

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

        model = xrv.models.get_model(weights="densenet121-res224-nih")
        model.op_threshs = None 
        model.classifier = torch.nn.Linear(1024,14) 
        # only training classifier
        optimizer = torch.optim.Adam(model.classifier.parameters())

        train_df, val_df, _ = handle_class_imbalance_df(data_path, logger)
        if args.test_mode:
                logger.warning('Using smaller dataset')
                train_subset_size = 100 
                val_subset_size = 50

                train_df = train_df.head(train_subset_size)
                val_df = val_df.head(val_subset_size)

        train_dataset = ChestXray14MultiClassDataset(dataframe=train_df, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

        val_dataset = ChestXray14MultiClassDataset(dataframe=val_df, transform=transform)
        validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

        if args.class_imbalance:
            logger.info('Handling class imbalance')
            class_weights = get_class_weights(train_df)
        else:
            class_weights = None

        trainer = TrainerMultiClass(
            model = model,
            class_weights=class_weights,
            model_output_folder = f'output/{args.output_folder}/model_checkpoints', 
            logger = logger,
            log_dir = f'runs/{args.output_folder}',
            optimizer = optimizer, 
        )
        trainer.train(
            train_dataloader = train_dataloader,
            validation_dataloader = validation_dataloader,
            num_epochs = args.num_epochs,
            idun_datetime_done = idun_datetime_done,
            model_arg = args.model
        )
    elif args.task == 'binary':
        dataloaders = BinaryClassificationDataLoader(
            data_path = data_path, 
            test_mode = args.test_mode, 
            batch_size = args.batch_size, 
            logger=logger, 
            test_size = 0.2, 
        )
        train_dataloader, validation_dataloader = dataloaders.get_dataloaders()
        model = DenseNetBinaryClassifier(logger=logger)
        model.log_params()

        trainer = TrainingModuleBinaryClass(
            model = model, 
            args = args, 
            logger = logger, 
            model_output_folder = f'output/{args.output_folder}/model_checkpoints', 
            output_folder = args.output_folder, 
            idun_time_done = idun_datetime_done
        )
        trainer.train_model(
            train_loader = train_dataloader, 
            validation_loader = validation_dataloader,
            num_epochs = args.num_epochs
        )

class DenseNetBinaryClassifier(nn.Module):
    def __init__(self, logger=None):
        super(DenseNetBinaryClassifier, self).__init__()
        self.logger=logger
        self.densenet = models.densenet121(weights=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.densenet(x)

    def log_params(self):
        total_params = sum(p.numel() for p in self.densenet.parameters())
        try: 
            self.logger.info(f"Total parameters in the model: {total_params}")
        except:
            print(f"Total parameters in the model:{total_params}")