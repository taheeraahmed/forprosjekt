from utils.set_up import set_up, str_to_bool
from dataloaders import MultiClassDataLoader, BinaryClassificationDataLoader
from models import DenseNetBinaryClassifier
from training_module import TrainingModuleBinaryClass, TrainingModuleMultiClass
import argparse
import sys

import torch
import torchxrayvision as xrv

def train(args):
    test_mode = args.test_mode
    model_arg = args.model

    setup_info = set_up(args)
    logger, idun_datetime_done, output_folder, model_output_folder = setup_info
    logger.info(f'Output folder: {output_folder}')

    data_path = '/cluster/home/taheeraa/datasets/chestxray-14'

    if test_mode:
        logger.warning(f'In test mode')
        args.num_epochs = 2
        args.batch_size = 3

    logger.info(f'batch_size: {args.batch_size}, num_epochs: {args.num_epochs}, lr: {args.learning_rate}')

    if model_arg == 'densenet-pretrained-xray-multi-class':
        dataloaders = MultiClassDataLoader(
            data_path=data_path, \
            test = test_mode, \
            batch_size = args.batch_size, \
            logger = logger, \
            test_size=0.01, \
            train_size=0.3 \
        )
        train_dataloader, validation_dataloader = dataloaders.get_dataloaders()

        model = xrv.models.get_model(weights="densenet121-res224-nih")
        model.op_threshs = None # prevent pre-trained model calibration
        model.classifier = torch.nn.Linear(1024,14) # reinitialize classifier
        
        trainer = TrainingModuleMultiClass(
            model=model, \
            logger=logger, \
            log_dir=f'runs/{args.output_folder}'
        )

        trainer.train(
            train_dataloader=train_dataloader, \
            validation_dataloader = validation_dataloader, \
            num_epochs= args.num_epochs, \
            idun_datetime_done = idun_datetime_done
        )
        
    elif model_arg == 'densenet-pretrained-imagenet-binary-class':
        dataloaders = BinaryClassificationDataLoader(data_path, test_mode, args.batch_size, logger, test_size=0.2, train_frac=0.25)
        train_loader, val_loader = dataloaders.get_dataloaders()

        model = DenseNetBinaryClassifier(logger=logger)
        model.log_params()

        logger.info('Started training')
        trainer = TrainingModuleBinaryClass(model, args, logger, model_output_folder, output_folder, idun_datetime_done,)
        trainer.train_model(train_loader, val_loader, args.num_epochs)
        
    else: 
        logger.error('Invalid model argument')
        sys.exit(1)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training with pytorch")
    parser.add_argument("-of", "--output_folder", help="Name of folder output files will be added", required=False, default='./output/')
    parser.add_argument("-it", "--idun_time", help="The duration of job set on IDUN", default=None, required=False)
    parser.add_argument("-t", "--test_mode", help="Test mode?", required=False, default=True)
    parser.add_argument("-m", "--model", choices=["densenet-pretrained-xray-multi-class", "densenet-pretrained-imagenet-binary-class"], help="Model to run", required=True)
    parser.add_argument("-e", "--num_epochs", help="Number of epochs", default=15)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", default=0.01)
    parser.add_argument("-bs", "--batch_size", help="Batch size", default=8)

    args = parser.parse_args()
    args.test_mode = str_to_bool(args.test_mode)
    train(args)