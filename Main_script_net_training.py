# %% Imports
from __future__ import print_function
import torch
import torch.optim as optim
import torchvision.transforms as tvtf
from datetime import datetime
import torchvision
import timeit
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys
import models
import predictor as pd
import csv
import pathlib
from pynvml import *
import threading
import time
import configparser
import psutil
import socket
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler


Force_GPU = None
if Force_GPU == None:
    nvmlInit()

# %% Definitions
root_dir = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'
root_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'Data'))+'//'
root_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'Data'))

root_dir = os.path.join(root_dir, '')
sys.path.append(str(os.path.dirname(__file__)))

def Write_to_trace_log(log_str, log_parameter= None):
    with open(os.path.join(os.getcwd(),"Learning_process_log.txt"), "a") as learning_logger:
        now = datetime.now()
        if log_parameter is None:
            learning_logger.write(f'{now}:: {log_str}\r\n')    
        else:
            learning_logger.write(f'{now}::  {log_str} : {log_parameter}\r\n')
            
def RunNet_NY_ClassBinary(classification_category='Left atrial enlargement'):
    print(f'NY Database classification, looking for {classification_category}')
    Write_to_trace_log('Starting learning process, category:',classification_category)
    import torch
    import torch.nn as nn
    import models
    import transforms as tf
    import matplotlib.pyplot as plt
    import NY_database_dataloader
    # torch.multiprocessing.freeze_support()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    print('Using device: ', device)
    Write_to_trace_log('Device',device)
    checkpoints_name = 'Ecg12LeadImageNet_NY_'+classification_category.replace(" ","_")
    ds = NY_database_dataloader.NY_Dataset(classification_category=classification_category,to_cut_image = True, \
        use_stored_data = False, stored_data_last_entries = 30)
    # for real training:
    num_train = int(len(ds)*0.8)  # 0.8
    num_val = int(len(ds)*0.05)
    num_test = len(ds) - num_train - num_val-1 

# %% Execution of the main loop
if __name__ == "__main__":
    print('Start execution')  
    classification_category='Atrial fibrillation'
    RunNet_NY_ClassBinary(classification_category=classification_category)
    print('Finished execution')



