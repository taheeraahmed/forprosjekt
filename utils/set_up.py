import pyfiglet
import logging
import os
from datetime import datetime
from utils.create_dir import create_directory_if_not_exists
from utils.check_gpu import check_gpu
import torch

def set_up():
    result = pyfiglet.figlet_format("Thoracic disease detection", font = "slant")  
    print(result) 
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    LOG_DIR = "/cluster/home/taheeraa/code/forprosjekt/runs"
    create_directory_if_not_exists(LOG_DIR)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    LOG_FILE = f"{LOG_DIR}/{dt_string}.txt"

    logging.basicConfig(level=logging.INFO, 
                format='[%(levelname)s] %(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(LOG_FILE),
                    logging.StreamHandler()
                ])
    logger = logging.getLogger()

    logger.info(f'Root directory of project: {project_root}')
    check_gpU(logger)
    logger.info('Set-up completed')
    return logger