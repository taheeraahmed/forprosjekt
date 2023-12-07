from utils.set_up import set_up, str_to_bool
from models.vit import vit
from models.densenet import densenet
import argparse
import sys

"""
TODO: Teste at alt funker!!
"""

def train(args):
    setup_info = set_up(args)
    logger, idun_datetime_done, output_folder = setup_info
    logger.info(f'Output folder: {output_folder}')
    data_path = '/cluster/home/taheeraa/datasets/chestxray-14'
    logger.info(f'batch_size: {args.batch_size}, num_epochs: {args.num_epochs}, lr: {args.learning_rate}')

    if args.model == 'vit':
        vit(
            logger=logger,
            args=args,
            idun_datetime_done=idun_datetime_done,
            data_path=data_path
        )
    elif args.model == 'densenet':
        densenet(
            logger=logger,
            args=args,
            idun_datetime_done=idun_datetime_done,
            data_path=data_path
        ) 
    else: 
        logger.error('Invalid model argument')
        sys.exit(1)

    logger.info('Training is done')
    
if __name__ == "__main__":
    model_choices = ['densenet','vit']
    task_choices = ['multi-class','binary']

    parser = argparse.ArgumentParser(description="Arguments for training with pytorch")
    # checkpoints, idun-time duration and test-mode
    parser.add_argument("-of", "--output_folder", help="Name of folder output files will be added", required=False, default='./output/')
    parser.add_argument("-it", "--idun_time", help="The duration of job set on IDUN", default=None, required=False)
    parser.add_argument("-t", "--test_mode", help="Test mode?", required=False, default=True)

    # model choices
    parser.add_argument("-m", "--model", choices=model_choices, help="Model to run", required=True)
    parser.add_argument("-im", "--class_imbalance", help="Handle class imbalance", required=False, default=False)
    parser.add_argument("-task", "--task", choices=task_choices, help="What task?", required=True)
    
    # model parameters
    parser.add_argument("-e", "--num_epochs", help="Number of epochs", type=int, default=15)
    parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=0.01)

    args = parser.parse_args()
    args.test_mode = str_to_bool(args.test_mode)
    args.class_imbalance = str_to_bool(args.class_imbalance)

    print(args.class_imbalance)
    print(type(args.class_imbalance))
    train(args)