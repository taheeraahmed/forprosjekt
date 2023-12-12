import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import ViTForImageClassification, ViTFeatureExtractor

from datasets.multi_class import ChestXray14MultiClassDataset
from trainers.trainer_multi_class import TrainerMultiClass
from utils.handle_class_imbalance import get_df_image_paths_labels, get_class_weights


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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    shuffle = True  
    num_workers = 4  

    train_df, val_df = get_df_image_paths_labels(args, data_path, logger)
    if args.test_mode:
        logger.warning('Using smaller dataset')
        train_subset_size = 100  # Adjust as needed
        val_subset_size = 50  # Adjust as needed

        train_df = train_df.head(train_subset_size)
        val_df = val_df.head(val_subset_size)

    train_dataset = ChestXray14MultiClassDataset(dataframe=train_df, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)

    val_dataset = ChestXray14MultiClassDataset(dataframe=val_df, transform=transform)
    validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=num_workers)
    
    if args.class_imbalance:
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