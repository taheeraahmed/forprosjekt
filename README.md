# Specialization project 

## Description

This repository contains a collection of notebooks and scripts focused on medical image analysis, particularly chest X-ray images. The code belongs to the pre-project for Computer Science TDT4501. 

The source code can be foudn in the `src` folder. The notebooks are located in the `notebooks` folder. 

## Installation

Before you get started, make sure you have installed Python 3.8 or higher, and have access to a GPU. 

Libraries listed in `requirements.txt` (e.g., pydicom, opencv-python, torch, etc.)

Additionally, you will need access to the relevant dataset: Chest X-Ray14. It can be downloaded from: https://www.kaggle.com/code/ingusterbets/nih-chest-x-rays-analysis. 

Also ensure you have downloaded these datasets and have them stored in an accessible location.

## Usage

There have been created two shell scripts of the root folder. These are `train-tf.sh` and `train.sh`. The first one is for training the model using TensorFlow, and the second one is for training the model using PyTorch. It has also been created for runninng through a slurm job, and is supposed to be ran on an HPC cluster (like IDUN). 

To run the scripts run the following commands in the terminal:
    
```bash
$ ./train-tf.sh
```

```bash
$ ./train.sh
```