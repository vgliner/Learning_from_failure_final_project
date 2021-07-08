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
import csv
import pathlib
from pynvml import *
import threading
import time
import configparser
import psutil
import socket
import loss_custom
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

def allocate_best_GPU(max_GPUs=None):
    import torch
    Recommended_gpu=-1
    if torch.cuda.is_available():
        GPUs_num = torch.cuda.device_count()
        if max_GPUs==None:
            max_GPUs=GPUs_num
        GPUs = []        
        for gpu_num in range(0,min(GPUs_num,max_GPUs)):  #1   
            h = nvmlDeviceGetHandleByIndex(gpu_num)
            info = nvmlDeviceGetMemoryInfo(h)
            # print(f'GPU: {gpu_num} , total    : {info.total}, free     : {info.free}, used     : {info.used}')
            GPUs.append(info.free)
        GPUs_r  = GPUs[::-1]
        Recommended_gpu = np.argmax(GPUs)# + 1
        Recommended_gpu = Recommended_gpu.item()
        print(f'Number of GPUs: {GPUs_num}, recommended GPU: {Recommended_gpu}')
    return Recommended_gpu

def Calc_sampler(origin_dataset, subset):
    origin_dataset.set_statistics_only(True)
    classifications_vec= []
    for indx in range(len(subset)):
        _,cl = subset[indx]
        classifications_vec.append(1.0) if cl else classifications_vec.append(0.0)
    class_counts = [sum(classifications_vec),len(classifications_vec)-sum(classifications_vec)] # Class True, Class False
    num_samples = len(classifications_vec)
    class_weights = [(num_samples-class_counts[0])/num_samples, class_counts[0]/num_samples ]
    weights = [class_weights[0] if classifications_vec[i] else class_weights[1] for i in range(num_samples)]
    samples_weight = torch.from_numpy(np.array(weights))    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)  
    origin_dataset.set_statistics_only(False)    
    return sampler

def RunNet_NY_Dual_Head(classification_categories=['Left atrial enlargement','Normal variant'], dropout = 0.26, batch_size = 70, label = 'XXX'):
    print(f'NY Database classification, looking for {classification_categories}')
    Write_to_trace_log('Starting learning process, category:',classification_categories)
    import torch
    import torch.nn as nn
    import models
    import transforms as tf
    import matplotlib.pyplot as plt
    import NY_database_dataloader
    # torch.multiprocessing.freeze_support()
    device = "cpu"
    if torch.cuda.is_available():
        # torch.cuda.empty_cache()
        if Force_GPU == None:
            GPU = allocate_best_GPU()
            # torch.cuda.device(GPU)
        else:
            GPU = Force_GPU
            torch.cuda.device(GPU)
        device = "cuda:"+str(GPU)
    print('Using device: ', device)
    Write_to_trace_log('Device',device)
    checkpoints_name = label+'Ecg12LeadImageNet_NY_'+'GPU'+str(GPU) #+classification_category.replace(" ","_")
    ds = NY_database_dataloader.NY_Dataset(classification_category=classification_categories,to_cut_image = True, \
        use_stored_data = False, stored_data_last_entries = 30, dual_class=True)
    # for real training:
    num_train = int(len(ds)*0.8)  # 0.8
    num_val = int(len(ds)*0.05)
    num_test = int((len(ds) - num_train - num_val-1 )*1.0)
    print(f'Using {num_train} entries for training, {num_val} for validation and {num_test} for test')

    batch_size = batch_size
    num_epochs = 100


    Write_to_trace_log('Batch size',batch_size)
    class_counts = ds.stats # Class True, Class False
    num_samples = sum(class_counts)


    # Dataloader training
    ds_train = tf.SubsetDataset(ds, num_train)  # (train=True, transform=tf_ds)
    # sampler = Calc_sampler(ds, ds_train) 
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False, num_workers=4, pin_memory=False)  # Without sampler  ,sampler = sampler
    x, y = next(iter(dl_train))
    print(f'Data shape is {np.shape(x)}, labels shape is {np.shape(y)}')
    in_h = x.shape[2]
    in_w = x.shape[3]
    in_channels = x.shape[1]

    # Validation dataset & loader
    ds_val = tf.SubsetDataset(ds, num_val, offset=num_train)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4, pin_memory=True)    

    # Test dataset & loader
    ds_test = tf.SubsetDataset(ds, num_test, offset=num_train + num_val)
    # sampler_t = Calc_sampler(ds, ds_test)    
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=False, num_workers=4, pin_memory=False)  #, sampler = sampler_t

    #%% Net structure
    hidden_channels = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [4] * 7 
    dropout = dropout
    stride = 2
    dilation = 1
    batch_norm = True
    fc_hidden_dims = [128]
    num_of_classes = 2
    model = models.Ecg12ImageNet(in_channels, hidden_channels, kernel_sizes, in_h, in_w,
                                 fc_hidden_dims, dropout=dropout, stride=stride,
                                 dilation=dilation, batch_norm=batch_norm, num_of_classes=2)#.to(device)
    model = model.to(device)                         

    # %% Test the dimentionality
    x_try = x.to(device, dtype=torch.float)
    y_pred = model(x_try)
    print('Output batch size is:',y_pred.shape[0], ', and number of class scores:', y_pred.shape[1], '\n')
    # num_correct = torch.sum((y_pred > 0).flatten() == (y.to(device, dtype=torch.long) == 1))
    # print(100*num_correct[0].item()/len(y),'% Accuracy of item 1 an ... maybe we should consider training the model')
    # print(f'First item accuracy is {100*num_correct[0].item()/y_pred.shape[0]} % , second is {100*num_correct[1].item()/y_pred.shape[0]} %... maybe we should consider training the model')
    # num_correct = torch.sum((y_pred > 0)== (y.to(device, dtype=torch.long) == 1), dim=0)
    del x, y, x_try, y_pred   

# %% Let's start training
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import torch.optim as optim
    from training import Ecg12LeadImageNetTrainerBinary
    torch.manual_seed(42)
    lr = 0.00003
    checkpoint_filename = f'{checkpoints_name}.pt'
    complete_path= os.path.join('checkpoints', checkpoint_filename)    
    # loss_fn = nn.BCEWithLogitsLoss() #  With weights for different classes, pos_weight>1 Increases the precision, < 1 the recall
    # loss_fn = nn.BCELoss()
    loss_fn = loss_custom.GeneralizedCELoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Ecg12LeadImageNetTrainerBinary(model, loss_fn, optimizer, device,optim_by_acc = False)
    fitResult = trainer.fit(dl_train, dl_test, num_epochs, checkpoints=complete_path,
                                early_stopping=100, print_every=1)


def RunLfFNet(classification_categories=['Left atrial enlargement','Normal variant'], dropout = 0.26, batch_size = 70, label = 'XXX'):
    print(f'Running reproduction of learning from failure net')
    import torch
    import torch.nn as nn
    import models
    import transforms as tf
    import matplotlib.pyplot as plt
    import NY_database_dataloader
    # torch.multiprocessing.freeze_support()
    device = "cpu"
    if torch.cuda.is_available():
        # torch.cuda.empty_cache()
        if Force_GPU == None:
            GPU = allocate_best_GPU()
            # torch.cuda.device(GPU)
        else:
            GPU = Force_GPU
            torch.cuda.device(GPU)
        device = "cuda:"+str(GPU)
    print('Using device: ', device)
    Write_to_trace_log('Device',device)    
    checkpoints_name = label+'LfF_'+'GPU'+str(GPU) #+classification_category.replace(" ","_")
    ds = NY_database_dataloader.NY_Dataset(classification_category=classification_categories,to_cut_image = True, \
        use_stored_data = False, stored_data_last_entries = 30, dual_class=True)
    # for real training:
    num_train = int(len(ds)*0.08)  # 0.8
    num_val = int(len(ds)*0.05)
    num_test = int((len(ds) - num_train - num_val-1 )*0.01)
    print(f'Using {num_train} entries for training, {num_val} for validation and {num_test} for test')

    batch_size = batch_size
    num_epochs = 100
    Write_to_trace_log('Batch size',batch_size)
    class_counts = ds.stats # Class True, Class False
    num_samples = sum(class_counts)
    # Dataloader training
    ds_train = tf.SubsetDataset(ds, num_train)  # (train=True, transform=tf_ds)
    # sampler = Calc_sampler(ds, ds_train) 
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False, num_workers=4, pin_memory=False)  # Without sampler  ,sampler = sampler
    x, y = next(iter(dl_train))
    print(f'Data shape is {np.shape(x)}, labels shape is {np.shape(y)}')
    in_h = x.shape[2]
    in_w = x.shape[3]
    in_channels = x.shape[1]

    # Validation dataset & loader
    ds_val = tf.SubsetDataset(ds, num_val, offset=num_train)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4, pin_memory=True)    

    # Test dataset & loader
    ds_test = tf.SubsetDataset(ds, num_test, offset=num_train + num_val)
    # sampler_t = Calc_sampler(ds, ds_test)    
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=False, num_workers=4, pin_memory=False)  #, sampler = sampler_t

    #%% Net structure
    hidden_channels = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [4] * 7 
    dropout = dropout
    stride = 2
    dilation = 1
    batch_norm = True
    fc_hidden_dims = [128]
    num_of_classes = 2
    model = models.Ecg12ImageNet(in_channels, hidden_channels, kernel_sizes, in_h, in_w,
                                 fc_hidden_dims, dropout=dropout, stride=stride,
                                 dilation=dilation, batch_norm=batch_norm, num_of_classes=2)#.to(device)
    model = model.to(device)                         


    model_biased = models.Ecg12ImageNet(in_channels, hidden_channels, kernel_sizes, in_h, in_w,
                                 fc_hidden_dims, dropout=dropout, stride=stride,
                                 dilation=dilation, batch_norm=batch_norm, num_of_classes=2)#.to(device)
    model_biased = model.to(device)         

    # %% Test the dimentionality
    x_try = x.to(device, dtype=torch.float)
    y_pred = model(x_try)
    print('Output batch size is:',y_pred.shape[0], ', and number of class scores:', y_pred.shape[1], '\n')
    # num_correct = torch.sum((y_pred > 0).flatten() == (y.to(device, dtype=torch.long) == 1))
    # print(100*num_correct[0].item()/len(y),'% Accuracy of item 1 an ... maybe we should consider training the model')
    # print(f'First item accuracy is {100*num_correct[0].item()/y_pred.shape[0]} % , second is {100*num_correct[1].item()/y_pred.shape[0]} %... maybe we should consider training the model')
    # num_correct = torch.sum((y_pred > 0)== (y.to(device, dtype=torch.long) == 1), dim=0)
    del x, y, x_try, y_pred   

# %% Let's start training
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import torch.optim as optim
    from training import Ecg12LeadImageNetTrainerBinary, LfFTrainer
    torch.manual_seed(42)
    lr = 0.0003
    checkpoint_filename = f'{checkpoints_name}.pt'
    complete_path= os.path.join('checkpoints', checkpoint_filename)    
    # loss_fn = nn.BCEWithLogitsLoss() #  With weights for different classes, pos_weight>1 Increases the precision, < 1 the recall
    # loss_fn = nn.BCELoss()
    # loss_fn = loss_custom.GeneralizedCELoss()
    loss_fn = nn.CrossEntropyLoss()
    biased_loss_fn = loss_custom.GeneralizedCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_biased = optim.Adam(model_biased.parameters(), lr=lr)
    trainer = LfFTrainer(model, loss_fn, optimizer, device,optim_by_acc = False, biased_model = model_biased,biased_loss_fn=biased_loss_fn, biased_optimizer=optimizer_biased)
    fitResult = trainer.fit(dl_train, dl_test, num_epochs, checkpoints=complete_path,
                                early_stopping=100, print_every=1)


# %% Execution of the main loop
if __name__ == "__main__":
    print('Start execution')  
    classification_categories=['Left ventricular hypertrophy','Normal variant']
    # """
    # ['Atrial fibrillation','Left ventricular hypertrophy','Normal variant']
    # """
    # RunNet_NY_Dual_Head(classification_categories=classification_categories, dropout = 0.26, batch_size = 82, label= 'ExpXXXX_')
    RunLfFNet(classification_categories=classification_categories, dropout = 0.1, batch_size = 110, label= 'Exp5_Overfit')
    print('Finished execution')



