import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import glob
from sklearn.metrics import classification_report
from utils.set_up import set_up

def train():
    logger = set_up()
    IMG_SIZE = [224, 224]
    BATCH_SIZE = 32

