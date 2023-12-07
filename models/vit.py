import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import ViTForImageClassification, ViTFeatureExtractor

from datasets import ChestXrayMutiClassDataset
from dataloaders import MultiClassDataLoader
from trainers.trainer_multi_class import TrainerMultiClass
from utils.handle_class_imbalance import handle_class_imbalance_df, get_class_weights


def vit(logger, args, idun_datetime_done, data_path):
    # TODO: Not handling binary, but don't need to do this anyway:)
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = torch.nn.Linear(model.classifier.in_features, 14)
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    shuffle = True  
    num_workers = 4  
    
    if args.class_imbalance:
        train_df, val_df, _ = handle_class_imbalance_df(data_path, logger)
        class_weights = get_class_weights(train_df)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        train_dataset = ChestXrayMutiClassDataset(dataframe=train_df, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

        val_dataset = ChestXrayMutiClassDataset(dataframe=val_df, transform=transform)
        validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

        trainer = TrainerMultiClass(
            model = model,
            class_weights=class_weights,
            model_output_folder = f'{args.output_folder}/model-checkpoints', 
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
    else: 
        dataloaders = MultiClassDataLoader(
            data_path= data_path, 
            test_mode = args.test_mode, 
            batch_size = args.batch_size, 
            logger = logger, 
            model_arg = args.model,
        )
        train_dataloader, val_dataloader = dataloaders.get_dataloaders()
        
        trainer = TrainerMultiClass(
            model = model, 
            model_output_folder = f'{args.output_folder}/model-checkpoints', 
            logger = logger, 
            log_dir=f'runs/{args.output_folder}',
            optimizer=optimizer,
            criterion=criterion,
        )
        trainer.train(
            train_dataloader = train_dataloader, 
            validation_dataloader = val_dataloader,
            num_epochs = args.num_epochs,
            idun_datetime_done = idun_datetime_done,
            model_arg = args.model_arg
        )