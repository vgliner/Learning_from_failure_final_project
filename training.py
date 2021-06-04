import os, sys

from numpy.core.numeric import True_
sys.path.append(str(os.path.dirname(__file__)))

from train_results import *
import abc
import os
import sys
import tqdm
import torch, gc
import pathlib

from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path
import numpy as np
from itertools import chain
from scipy.integrate import simps
import re
import matplotlib.pyplot as plt
from sklearn import metrics
import time


def ROC_AUC(net_output,absolute_truth):
    normalized_output = torch.cat(net_output)
    abs_truth = torch.cat(absolute_truth)
    try:
        if np.sum(np.abs(np.diff(abs_truth.detach().cpu().numpy()))) == 0.0:
            return 0.0        
        Integ = metrics.roc_auc_score(abs_truth.detach().cpu().numpy(), normalized_output.detach().cpu().numpy()) #, pos_label=1.0
    except:
        return 0.0
    if Integ == None:
        return 0.0
    return Integ


def Find_optimal_threshold(net_output,absolute_truth):
    # print(type(net_output))
    # print(f'Net output is: {net_output}')
    normalized_output = torch.sigmoid(torch.from_numpy(np.array(net_output)))
    abs_truth = torch.cat(absolute_truth)
    threshold_step= 0.02
    threshold = -1.0
    accuracy_threshold = -1.0
    best_accuracy = 0.0
    F1_threshold = 0.0
    best_F1 = 0.0
    normalized_output= normalized_output.view(1,-1).detach().cpu().numpy()
    abs_truth = abs_truth.detach().cpu().numpy()
    while (threshold<=1.0):
        TPs = np.sum((normalized_output> threshold)*(abs_truth == 1.0))
        TNs = np.sum((normalized_output<=threshold)*(abs_truth == 0.0))
        FPs = np.sum((normalized_output> threshold)*(abs_truth == 0.0))
        FNs = np.sum((normalized_output<=threshold)*(abs_truth == 1.0))
        accuracy = float(TPs+TNs)/float(TPs+TNs+FPs+FNs)
        F1 = float(TPs)/float(TPs+0.5*(FPs+FNs))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            accuracy_threshold = threshold
        if F1 > best_F1:
            best_F1 = F1
            F1_threshold = threshold
        threshold += threshold_step

    return (accuracy_threshold,F1_threshold)

class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer,
        device='cuda',classification_threshold=None,
        optim_by_acc= True, multiple_heads=False, loss_fn_domain=None,
        num_of_epochs=None, epoch_to_start_disturbing= 0,lr_factor_between_heads = 10.0 ):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        :param optim_by_acc: Choice to optimize either by ROC or accuracy
        :param multiple_heads: Choice to optimize adversarial nets
        """
        self.model = model
        self.loss_fn = loss_fn
        self.loss_fn_source = loss_fn
        self.loss_fn_domain = loss_fn_domain
        self.optimizer = optimizer
        self.device = device
        self.classification_threshold = classification_threshold
        self.optim_by_acc = optim_by_acc
        self.multiple_heads = multiple_heads
        self.current_epoch = 0
        self.num_of_epochs = num_of_epochs
        try:
            self.grl_lambda =model.grl_lambda
        except:
            self.grl_lambda = 1
        self.epoch_to_start_disturbing=epoch_to_start_disturbing
        self.num_of_epochs_without_improvement = 0
        self.former_acc = 0.0
        self.max_acc_so_far = 0.0
        self.lr_factor_between_heads = lr_factor_between_heads
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, dl_target: DataLoader = None,**kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set. For adversarial use it as a source
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :param dl_target: Dataloader for the training set, specific for adversarial. Use it as a target
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc, TPs, TNs, FPs, FNs, Is_checkpointed = [], [], [], [], [], [], [], [], []
        ROC_AUCs= []
        epochs_without_improvement = 0
        best_acc = 0.0
        best_ROC_AUC = 0.0
        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = checkpoints
            dir_path = os.path.dirname(os.path.realpath(__file__))
            complete_path= os.path.join(dir_path,checkpoint_filename)
            if os.path.isfile(complete_path):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(complete_path,
                                         map_location=self.device)
                try:
                    best_acc = saved_state.get('best_acc', best_acc)
                except:
                    print('No best accuracy saved in the model')
                try:
                    best_ROC_AUC = saved_state.get('best_ROC_AUC', best_ROC_AUC)
                except:
                    print('No best ROC AUC saved in the model')                    

                epochs_without_improvement =\
                    saved_state.get('ewi', epochs_without_improvement)
                try:
                    self.model = saved_state['full_model']
                except:
                    print('Didnt find saved full model')
                self.model.load_state_dict(saved_state['model_state'])
        checkpoints_name_parsed = re.split(r'\\|/',checkpoints)
        with open(f"Execution_dump_kernel_{checkpoints_name_parsed[-1][:-3]}.txt", "a") as myfile:
            myfile.write(f'EPOCH \t TR_ACC \t TE_ACC \t TR_LOSS \t TE_LOSS_t \t IS_BEST\n')
            
        for epoch in range(num_epochs):
            gc.collect()    
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            (loss, acc,TP, TN, FP, FN, out, y, ROC_AUC_) = train_result
            train_loss += loss
            train_acc.append(acc)
            tr_acc=round(acc,2)
            tr_loss=loss[-1]            
            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            (loss, acc,TP, TN, FP, FN, out, y,ROC_AUC_) = test_result
            TPs.append(TP)
            TNs.append(TN)
            FPs.append(FP)
            FNs.append(FN)
            te_acc=round(acc,2)
            te_loss=loss[-1]       
            test_loss += loss
            # out1 = np.array([sub.detach().cpu().numpy() for sub in out ])  #for j in sub
            # y = np.array([sub.cpu().detach().numpy() for sub in y]) # for j in sub
            # out = out.astype('float64')
            # y = y.astype('float64')
            # ROC_area_under_curve = self.ROC_AUC(out,y)
            # ROC_AUCs.append(ROC_AUC_)
            # Adding additional printout 
            if verbose:
                precision = 0
                recall = 0
                if (TP.item()+FP.item())>0:
                    precision = round(TP.item()/ (TP.item()+FP.item()),2)
                if (TP.item()+FN.item())>0:
                    recall = round(TP.item()/(TP.item()+FN.item()),2)
                print(f'ACC: {acc}, PRECISION: {precision}, RECALL: {recall} COMBINED: {(precision+recall)/2}')
            try:
                # if self.optim_by_acc:
                is_best=(te_acc > best_acc)
                # else:
                #     is_best=(ROC_area_under_curve > best_ROC_AUC)
            except:
                is_best=False
            with open(f"Execution_dump_kernel_{checkpoints_name_parsed[-1][:-3]}.txt", "a") as myfile:
                myfile.write(f'{actual_num_epochs} \t {tr_acc} \t {te_acc} \t {tr_loss} \t {te_loss} \t {is_best} \n')
            if True:   #is_best
                with open(f"Results_raw_dump_kernel_{checkpoints_name_parsed[-1][:-3]}.txt", "a") as myfile:
                    myfile.write(f'{actual_num_epochs} \t {te_acc} \t {te_loss} \t {is_best} \t {TP.item()} \t {TN.item()} \t {FP.item()} \t {FN.item()} \n') #{ROC_AUC_}\n
            actual_num_epochs += 1

            if checkpoints:
                if not best_acc:
                    best_acc = acc
                if acc > best_acc:
                    best_acc = acc
                    if self.optim_by_acc:
                        save_checkpoint = True
                # if not best_ROC_AUC:
                #     best_ROC_AUC = ROC_area_under_curve
                # if ROC_area_under_curve > best_ROC_AUC:
                #     best_ROC_AUC = ROC_area_under_curve
                    if self.optim_by_acc == False:
                        save_checkpoint = True
            if test_acc:
                if self.optim_by_acc:
                    if acc <= test_acc[-1]:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0
                else:
                    # if ROC_area_under_curve <= ROC_AUCs[-1]:
                    #     epochs_without_improvement += 1
                    # else:
                    #     epochs_without_improvement = 0          
                    pass          

            test_acc.append(acc)

            if early_stopping:
                if epochs_without_improvement >= early_stopping:
                    break

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                sample_dims = np.shape(dl_train.dataset[0][0])
                # accuracy_threshold,F1_threshold = Find_optimal_threshold(out,y)
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict(),
                                   full_model= self.model, best_ROC_AUC=best_ROC_AUC,
                                   sample_dims=sample_dims)
                dir_path = os.path.dirname(os.path.realpath(__file__))
                complete_path= os.path.join(dir_path,checkpoint_filename)
                torch.save(saved_state,complete_path)
                Is_checkpointed.append(True)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch+1}')
            else:
                Is_checkpointed.append(False)

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc, TPs, TNs, FPs, FNs, Is_checkpointed, ROC_AUCs)


    def fit_complex_net(self, dl_source_train: DataLoader, dl_source_test: DataLoader,
            dl_target_train: DataLoader, dl_target_test: DataLoader, num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None,**kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_source_train: Dataloader for the source training set
        :param dl_source_test: Dataloader for the source test set.
        :param dl_target_train: Dataloader for the target training set
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        try:
            workers_num = kw['workers_num']
        except:
            workers_num = 8
        optimize_by_source = True
        grl_lambda = kw['grl_lambda']
        device = kw['device']
        # Note=kw['Note']
        # print(f'grl_lambda= {grl_lambda}')
        actual_num_epochs = 0
        train_loss, train_acc_s, train_acc_t, test_loss, test_acc_s, test_acc_t,TPs, TNs, FPs, FNs,TPt, TNt, FPt, FNt, Is_checkpointed = [], [], [], [], [], [], [], [], [], [],[], [], [], [], []
        ROC_AUCs,ROC_AUCt, test_acc_targets= [],[],[]
        epochs_without_improvement = 0
        best_acc_s = 0.0
        best_acc_t = 0.0
        best_test_loss = 10000.0
        best_ROC_AUC_s = 0.0
        best_ROC_AUC_t = 0.0
        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = checkpoints
            dir_path = os.path.dirname(os.path.realpath(__file__))
            complete_path= os.path.join(dir_path,checkpoint_filename)
            upload_checkpoint = complete_path[:-3]+'_acc_s.pt'
            try:
                if os.path.isfile(upload_checkpoint):
                    print(f'*** Loading checkpoint file {checkpoint_filename}')
                    saved_state = torch.load(upload_checkpoint,
                                            map_location=self.device)
                    try:
                        best_acc_s = saved_state.get('best_acc_s', best_acc_s)
                        best_acc_t = saved_state.get('best_acc_t', best_acc_t)
                    except:
                        print('No best accuracy saved in the model')
                    try:
                        best_ROC_AUC_s = saved_state.get('best_ROC_AUC_s', best_ROC_AUC_s)
                        best_ROC_AUC_t = saved_state.get('best_ROC_AUC_t', best_ROC_AUC_t)
                    except:
                        print('No best ROC AUC saved in the model')                    

                    epochs_without_improvement =saved_state.get('ewi', epochs_without_improvement)
                    try:
                        self.model = saved_state['full_model']
                    except:
                        print('Didnt find saved full model')
                    self.model.load_state_dict(saved_state['model_state'])
            except:
                print('Failed to find checkpoints folder')                

    ##LOGGER TITLES                
        checkpoints_name_parsed = re.split(r'\\|/',checkpoints)
        log_file_path=os.path.join(os.getcwd(), 'Logs')
        if not os.path.exists(log_file_path):
            print('Log files folder does not exist, creating...')
            os.makedirs(log_file_path)            
        execution_dump_path= os.path.join(log_file_path,f"Execution_dump_kernel_{checkpoints_name_parsed[-1][:-3]}_lr_{round(self.get_lr(),8)}.txt")
        results_raw_dump_path = os.path.join(log_file_path,f"Results_raw_dump_kernel_{checkpoints_name_parsed[-1][:-3]}_lr_{round(self.get_lr(),8)}.txt")
        with open(execution_dump_path, "w") as myfile:
            myfile.write(f'EPOCH \t TR_LOSS_s \t TR_LOSS_t \t TEST_LOSS_s \t TEST_LOSS_t \t TR_ACC_SRC \t TEST_ACC_SRC \t'
                f'IS_BEST_SRC_ACC\t IS_BEST_SRC_ROC\t TR_ACC_TGT \t TEST_ACC_TGT \t IS_BEST_TGT_ACC\t'
                f'IS_BEST_TGT_ROC\tLAMBDA\n')
        if False:   #is_best
            with open(results_raw_dump_path, "w") as myfile:
                myfile.write(f'EPOCH \t TR_LOSS_s \t TR_LOSS_t \t TEST_LOSS_s \tTEST_LOSS_t \t TR_ACC_SRC \t TEST_ACC_SRC \t'
                    f'IS_BEST_SRC_ACC\t IS_BEST_SRC_ROC\t TR_ACC_TGT \t TEST_ACC_TGT \t IS_BEST_TGT_ACC\t'
                    f'IS_BEST_TGT_ROC \t TP_s \t TN_s \t FP_s \t FN_s \t TP_t \t TN_t \t FP_t \t FN_t'
                    f'ROC_AUCs \t ROC_AUCt\tLAMBDA\n')                
        for epoch in range(num_epochs):
            gc.collect()    
            epoch_start = time.time()
            save_checkpoint_s = False
            save_checkpoint_t = False
            save_checkpoint_s_ROC = False
            save_checkpoint_t_ROC = False            
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)
            # self.epoch = epoch
            # self.num_epochs = num_epochs
            kw['Current_epoch_num'] = epoch
            train_result = self.train_epoch_complex_net(dl_source_train,dl_target_train,
                verbose=verbose,epoch= epoch,num_epochs=num_epochs,  **kw)
            (loss_s,loss_t,  acc_s,  acc_t,TP_s, TN_s, FP_s, FN_s,TP_t, TN_t, FP_t, FN_t,
             out_s, y_s, out_t, y_t,grl_lambda_train) = train_result
            train_loss.append([loss_s.item(),loss_t.item()])
            tr_acc_s=round(acc_s,2)
            train_acc_s.append(tr_acc_s)
            tr_acc_t=round(acc_t,2)            
            train_acc_t.append(tr_acc_t)
            tr_loss_s=round(loss_s.item(),5)
            tr_loss_t=round(loss_t.item(),5)
            test_result = self.test_epoch_complex_net(dl_source_test, dl_target_test,
                verbose=verbose, epoch= epoch,num_epochs=num_epochs,  **kw)
            (loss_s,loss_t, acc_s,  acc_t,TP_s, TN_s, FP_s, FN_s,TP_t, TN_t, FP_t, FN_t,
             out_s, y_s, out_t, y_t,_) = test_result
            TPs.append(TP_s)
            TNs.append(TN_s)
            FPs.append(FP_s)
            FNs.append(FN_s)
            TPt.append(TP_t)
            TNt.append(TN_t)
            FPt.append(FP_t)
            FNt.append(FN_t)            
            te_acc_s=round(acc_s,2)
            te_acc_t=round(acc_t,2)
            test_acc_targets.append(te_acc_t)
            te_loss_s=np.mean(loss_s)
            te_loss_t=np.mean(loss_t)
            ROC_area_under_curve_s = round(self.ROC_AUC(out_s,y_s),2)
            ROC_area_under_curve_t = round(self.ROC_AUC(out_t,y_t),2)
            if (ROC_area_under_curve_s>1) or (ROC_area_under_curve_t>1) or (ROC_area_under_curve_s<0) or (ROC_area_under_curve_t<0):
                print('Found problematic ROC')
            ROC_AUCs.append(ROC_area_under_curve_s)
            ROC_AUCt.append(ROC_area_under_curve_t)
            # Adding additional printout 
            if verbose:
                precision_s = 0
                recall_s = 0
                precision_t = 0
                recall_t = 0                
                if (TP_s.item()+FP_s.item())>0:
                    precision_s = round(TP_s.item()/ (TP_s.item()+FP_s.item()),2)
                if (TP_s.item()+FN_s.item())>0:
                    recall_s = round(TP_s.item()/(TP_s.item()+FN_s.item()),2)
                if (TP_t.item()+FP_t.item())>0:
                    precision_t = round(TP_t.item()/ (TP_t.item()+FP_t.item()),2)
                if (TP_t.item()+FN_t.item())>0:
                    recall_t = round(TP_t.item()/(TP_t.item()+FN_t.item()),2)                    
                print(f'SOURCE: ACC TR: {tr_acc_s}, ACC TEST: {te_acc_s}, LOSS: {round(te_loss_s,5)}, PRECISION: {precision_s}, RECALL: {recall_s}, ROC: {ROC_area_under_curve_s}')
                print(f'TARGET: ACC TR: {tr_acc_t}, ACC TEST: {te_acc_t}, LOSS: {round(te_loss_t,5)}, PRECISION: {precision_t}, RECALL: {recall_t}, ROC: {ROC_area_under_curve_t}')

            is_best_s=(te_acc_s > best_acc_s) and (epoch > self.epoch_to_start_disturbing+10) and (np.abs(te_acc_t-50.0)<15) and (te_loss_s< best_test_loss)
            if is_best_s:
                best_acc_s = te_acc_s
                best_test_loss = te_loss_s 
            is_best_t=False #(te_acc_t > best_acc_t)
            if is_best_t:
                best_acc_t = te_acc_t
            is_best_ROC_s=(ROC_area_under_curve_s > best_ROC_AUC_s)  and (epoch > self.epoch_to_start_disturbing+10) and (np.abs(te_acc_t-50.0)<15) and (te_loss_s< best_test_loss)
            if is_best_ROC_s:
                best_ROC_AUC_s = ROC_area_under_curve_s
                best_test_loss = te_loss_s 
            is_best_ROC_t=False #(ROC_area_under_curve_t > best_ROC_AUC_t)
            if is_best_ROC_t:
                best_ROC_AUC_t = ROC_area_under_curve_t

            with open(execution_dump_path, "a") as myfile:
                myfile.write(f'{actual_num_epochs}\t{tr_loss_s}\t{tr_loss_t}\t{te_loss_s}\t{te_loss_t}\t{tr_acc_s}\t{te_acc_s}\t{is_best_s}\t{is_best_ROC_s}\t{tr_acc_t}\t{te_acc_t}\t{is_best_t}\t{is_best_ROC_t}\t{grl_lambda_train}\n')
            if False:   #is_best
                with open(results_raw_dump_path, "a") as myfile:
                    myfile.write(f'{actual_num_epochs} \t {tr_loss_s}\t {tr_loss_t} \t {te_loss_s} \t {te_loss_t} \t{tr_acc_s} \t {te_acc_s} \t \
                    {is_best_s} \t {is_best_ROC_s} \t {tr_acc_t} \t {te_acc_t} \t {is_best_t} \t {is_best_ROC_t} \t \
                    {TP_s.item()} \t {TN_s.item()} \t {FP_s.item()} \t {FN_s.item()} \t \
                    {TP_t.item()} \t {TN_t.item()} \t {FP_t.item()} \t {FN_t.item()} \t \
                    {ROC_area_under_curve_s} \t {ROC_area_under_curve_t} \t {grl_lambda_train}\n')                    
            actual_num_epochs += 1
            self.current_epoch =  actual_num_epochs           
            if checkpoints:
                if epoch > self.epoch_to_start_disturbing:
                    if is_best_s:
                        save_checkpoint_s = True
                    # if is_best_t:
                    #     save_checkpoint_t = True            
                    if is_best_ROC_s:
                        save_checkpoint_s_ROC = True
                    # if is_best_t:
                    #     save_checkpoint_t_ROC = True   
                    if save_checkpoint_s or save_checkpoint_t or save_checkpoint_s_ROC or save_checkpoint_t_ROC:
                        Is_checkpointed.append(True)
                else:
                    Is_checkpointed.append(False)
                   

            if len(test_acc_s):
                if self.optim_by_acc:
                    if acc_s <= test_acc_s[-1]:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0
                else:
                    if ROC_area_under_curve_s <= ROC_AUCs[-1]:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0                    

            test_acc_s.append(acc_s)

            if early_stopping:
                if epochs_without_improvement >= early_stopping:
                    break
            
            accuracy_threshold,F1_threshold = Find_optimal_threshold(out_s,y_s)            
            if save_checkpoint_s and checkpoint_filename is not None: # Save model checkpoint if requested ACC. SOURCE
                sample_dims = np.shape(dl_source_train.dataset[0][0])
                accuracy_threshold,F1_threshold = Find_optimal_threshold(out_s,y_s)
                saved_state = dict(te_acc_s=te_acc_s,te_acc_t=te_acc_t,
                                ROC_area_under_curve_s=ROC_area_under_curve_s,ROC_area_under_curve_t=ROC_area_under_curve_t,
                                ewi=epochs_without_improvement,model_state=self.model.state_dict(),
                                full_model= self.model,accuracy_threshold=accuracy_threshold,
                                F1_threshold=F1_threshold,sample_dims=sample_dims)
                dir_path = os.path.dirname(os.path.realpath(__file__))
                fn = checkpoint_filename[:-3]+'_acc_s'+checkpoint_filename[-3:]
                complete_path= os.path.join(dir_path,fn)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)                
                torch.save(saved_state,complete_path)
                Is_checkpointed.append(True)
                print(f'*** Saved checkpoint ACC source {fn} at epoch {epoch+1}')
            if save_checkpoint_t and checkpoint_filename is not None: # Save model checkpoint if requested ACC. TARGET
                sample_dims = np.shape(dl_source_train.dataset[0][0])
                accuracy_threshold,F1_threshold = Find_optimal_threshold(out_s,y_s)
                saved_state = dict(te_acc_s=te_acc_s,best_acc_t=best_acc_t,
                                ROC_area_under_curve_s=ROC_area_under_curve_s,ROC_area_under_curve_t=ROC_area_under_curve_t,
                                ewi=epochs_without_improvement,model_state=self.model.state_dict(),
                                full_model= self.model,accuracy_threshold=accuracy_threshold,
                                F1_threshold=F1_threshold,sample_dims=sample_dims)
                dir_path = os.path.dirname(os.path.realpath(__file__))
                fn = checkpoint_filename[:-3]+'_acc_t'+checkpoint_filename[-3:]
                complete_path= os.path.join(dir_path,fn)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)                        
                torch.save(saved_state,complete_path)
                Is_checkpointed.append(True)
                print(f'*** Saved checkpoint ACC target {fn} at epoch {epoch+1}')   
            if save_checkpoint_s_ROC and checkpoint_filename is not None: # Save model checkpoint if requested ROC. SOURCE
                sample_dims = np.shape(dl_source_train.dataset[0][0])
                accuracy_threshold,F1_threshold = Find_optimal_threshold(out_s,y_s)
                saved_state = dict(te_acc_s=te_acc_s,best_acc_t=best_acc_t,
                                ROC_area_under_curve_s=ROC_area_under_curve_s,ROC_area_under_curve_t=ROC_area_under_curve_t,
                                ewi=epochs_without_improvement, model_state=self.model.state_dict(),
                                full_model= self.model,accuracy_threshold=accuracy_threshold,
                                F1_threshold=F1_threshold,sample_dims=sample_dims)
                dir_path = os.path.dirname(os.path.realpath(__file__))
                fn = checkpoint_filename[:-3]+'_ROC_s'+checkpoint_filename[-3:]
                complete_path= os.path.join(dir_path,fn)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)                        
                torch.save(saved_state,complete_path)
                Is_checkpointed.append(True)
                print(f'*** Saved checkpoint ROC source {fn} at epoch {epoch+1}') 
            if save_checkpoint_t_ROC and checkpoint_filename is not None: # Save model checkpoint if requested ROC. TARGET
                sample_dims = np.shape(dl_source_train.dataset[0][0])
                accuracy_threshold,F1_threshold = Find_optimal_threshold(out_s,y_s)
                saved_state = dict(te_acc_s=te_acc_s,best_acc_t=best_acc_t,
                                ROC_area_under_curve_s=ROC_area_under_curve_s,ROC_area_under_curve_t=ROC_area_under_curve_t,
                                ewi=epochs_without_improvement, model_state=self.model.state_dict(),
                                full_model= self.model,accuracy_threshold=accuracy_threshold,
                                F1_threshold=F1_threshold,sample_dims=sample_dims)
                dir_path = os.path.dirname(os.path.realpath(__file__))
                fn = checkpoint_filename[:-3]+'_ROC_t'+checkpoint_filename[-3:]
                complete_path= os.path.join(dir_path,fn)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)                        
                torch.save(saved_state,complete_path)
                Is_checkpointed.append(True)
                print(f'*** Saved checkpoint ROC target {fn} at epoch {epoch+1}') 

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

            epoch_stop = time.time()
            print(f'Epoch elapsed time: {round(epoch_stop-epoch_start,2)} sec,{round((epoch_stop-epoch_start)/60,2)} min.')
            dl_source_train.num_workers = workers_num
            dl_source_test.num_workers = workers_num
            dl_target_train.num_workers = workers_num
            dl_target_test.num_workers = workers_num

        return FitResult_complex_net(num_epochs=actual_num_epochs,train_loss=train_loss,
                         train_acc_source=train_acc_s, train_acc_target=train_acc_t, test_loss_source=None,
                         test_acc_source = test_acc_s,test_loss_target=test_loss, test_acc_target=test_acc_targets,
                         TPs_source= TPs,TNs_source= TNs, 
                         FPs_source=FPs,FNs_source= FNs,TPs_target=TPt,TNs_target=TNt,
                         FPs_target=FPt,FNs_target=FNt, Is_checkpointed= Is_checkpointed,ROC_AUCs= ROC_AUCs,
                         ROC_AUCt=ROC_AUCt)


    def ROC_AUC(self,net_output,absolute_truth):
        normalized_output = np.concatenate(net_output, axis=0)
        abs_truth = torch.cat(absolute_truth)
        abs_truth = abs_truth.detach().cpu().numpy()
        try:
            if np.sum(abs_truth) == 0.0:
                return 0.0        
            Integ = metrics.roc_auc_score(abs_truth, normalized_output) #, pos_label=1.0
        except:
            return 0.0
        if Integ == None:
            return 0.0
        return Integ

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def train_epoch_complex_net(self, dl_source_train: DataLoader, dl_target_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch_complex_net(dl_source_train,dl_target_train, self.train_batch, **kw)        
 
    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation test mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def test_epoch_complex_net(self, dl_source_test: DataLoader, dl_target_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation test mode
        return self._foreach_batch_complex_net(dl_source_test,dl_target_test, self.test_batch, **kw)        

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        y= []
        out = []
        num_correct = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)
                y.append(batch_res.y)
                out.append(batch_res.out)
                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
                TP += batch_res.num_TP
                TN += batch_res.num_TN
                FP += batch_res.num_FP
                FN += batch_res.num_FN

            ROC_AUC_ = ROC_AUC(out,y)
            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy,num_TP=TP,num_TN=TN,num_FP=FP,num_FN=FN, y=y, out = out, ROC_AUC = ROC_AUC_)

    @staticmethod
    def _grl_lambda_calc(batch_idx : int ,epoch_idx : int,num_batches :int,n_epochs : int):    # Custom grl_lambda
        p = float(batch_idx + epoch_idx * num_batches) / (n_epochs * num_batches)
        # grl_lambda = 5.0/(1+np.exp(-p))-1
        # if epoch_idx< 30:
        #     grl_lambda = 0.0
        # else:
        #     grl_lambda = 1.0    
        # grl_lambda = 0.2 * p -0.1       
        # grl_lambda = 6.0/(1.0+np.exp(-3*p))-1
        grl_lambda = 0.85 * np.exp(5.5*p)
        return grl_lambda



    @staticmethod
    def _foreach_batch_complex_net(dl_source: DataLoader, dl_target: DataLoader, 
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None, **kw) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses_s = []
        losses_t = []
        y_s= []
        y_t= []
        out_s = []
        out_t = []
        num_correct_s = 0
        num_correct_t = 0
        TP_s = 0
        TN_s = 0
        FP_s = 0
        FN_s = 0
        TP_t = 0
        TN_t = 0
        FP_t = 0
        FN_t = 0        
        num_batches = min(len(dl_source), len(dl_target))-1
        num_samples = num_batches * dl_source.batch_size
        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl_source.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_source_iter = iter(dl_source)
            dl_target_iter = iter(dl_target) 
            epoch_idx = kw['epoch']
            n_epochs = kw['num_epochs']
            debug_logger_s =[]
            debug_logger_t =[]
            grl_lambda_kw = kw['grl_lambda']
            for batch_idx in range(num_batches):
                # Training progress and GRL lambda
                if grl_lambda_kw == -6.0:
                ##############################################
                    grl_lambda = Trainer._grl_lambda_calc(batch_idx,epoch_idx,num_batches,n_epochs)
                else:
                    grl_lambda = grl_lambda_kw
                ##############################################
                
                # Train on source domain
                data_source = next(dl_source_iter)
                data_target = next(dl_target_iter)

                # y_s_domain = torch.zeros(len(y_s))

                batch_res = forward_fn(data_source,data_target, grl_lambda, epoch_num = epoch_idx)
                y_s.append(batch_res.y_source)
                y_t.append(batch_res.y_target)
                out_s.append(batch_res.out_source)
                out_t.append(batch_res.out_target)
                losses_s.append(batch_res.loss_s)
                losses_t.append(batch_res.loss_t)
                num_correct_s += batch_res.num_correct_source
                num_correct_t += batch_res.num_correct_target
                debug_logger_s.append(batch_res.num_correct_source)
                debug_logger_t.append(batch_res.num_correct_target)
                TP_s += batch_res.num_TP_source
                TN_s += batch_res.num_TN_source
                FP_s += batch_res.num_FP_source
                FN_s += batch_res.num_FN_source
                TP_t += batch_res.num_TP_target
                TN_t += batch_res.num_TN_target
                FP_t += batch_res.num_FP_target
                FN_t += batch_res.num_FN_target
                pbar.set_description(f'{pbar_name} ({batch_res.loss_s:.3f},{batch_res.loss_t:.3f})')
                pbar.update()


            out_s = [torch.sigmoid(l).detach().cpu().numpy() for l in out_s]
            out_t = [torch.sigmoid(l).detach().cpu().numpy() for l in out_t]
            avg_loss_s = sum(losses_s) / num_batches
            avg_loss_t = sum(losses_t) / num_batches
            accuracy_s = 100. * num_correct_s / num_samples
            accuracy_t = 100. * num_correct_t / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss_s:.3f},{avg_loss_t:.3f}, '
                                 f'Accuracy source :{accuracy_s:.1f},'
                                 f'Accuracy target :{accuracy_t:.1f}),')

        return EpochResult_complex_net(losses_s=avg_loss_s,losses_t=avg_loss_t,
            accuracy_source=accuracy_s,accuracy_target=accuracy_t, num_TP_source=TP_s,num_TN_source=TN_s,
            num_FP_source=FP_s,num_FN_source=FN_s,num_TP_target=TP_t,num_TN_target=TN_t,
            num_FP_target= FP_t,num_FN_target= FN_t,out_source= out_s, y_source = y_s, 
            out_target= out_t,y_target=y_t,grl_lambda=grl_lambda)


class Ecg12LeadNetTrainerBinary(Trainer):

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x).flatten()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        num_correct = torch.sum((out > 0) == (y == 1))
        TP = torch.sum((out > 0) * (y == 1))
        TN = torch.sum((out <= 0) * (y == 0))
        FP = torch.sum((out > 0) * (y == 0))
        FN = torch.sum((out <= 0) * (y == 1))

        return BatchResult(loss.item(), num_correct.item(),TP,TN,FP,FN,out,y)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x).flatten()
            loss = self.loss_fn(out, y)
            num_correct = torch.sum((out > 0) == (y == 1))
            out_norm=torch.sigmoid(out)
            if self.classification_threshold==None:
                TP = torch.sum((out > 0) * (y == 1))
                TN = torch.sum((out <= 0) * (y == 0))
                FP = torch.sum((out > 0) * (y == 0))
                FN = torch.sum((out <= 0) * (y == 1))
            else:
                TP = torch.sum((out_norm >= self.classification_threshold) * (y == 1))
                TN = torch.sum((out_norm < self.classification_threshold) * (y == 0))
                FP = torch.sum((out_norm >= self.classification_threshold) * (y == 0))
                FN = torch.sum((out_norm < self.classification_threshold) * (y == 1))  
                num_correct = torch.sum((out_norm > self.classification_threshold) == (y == 1))


        return BatchResult(loss.item(), num_correct.item(),TP,TN,FP,FN, out, y)


class Ecg12LeadNetTrainerMulticlass(Trainer):

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        self.optimizer.zero_grad()

        out = self.model(x)#.flatten()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()
        
        indices = out>0#torch.max(out, 1)  #_, 
        indices1 = y>0 #torch.max(y, 1)  #_, 

        num_correct = torch.sum(indices==indices1)

        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = (x[0].to(self.device, dtype=torch.float), x[1].to(self.device, dtype=torch.float))
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out.flatten(), y.flatten())
            indices = out>0 #torch.max(out, 1) _, 
            indices1 = y>0 #torch.max(y, 1) _, 

            num_correct = torch.sum(indices==indices1)
        return BatchResult(loss.item(), num_correct.item())

class Ecg12LeadImageNetTrainerBinary(Trainer): #

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        # x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float)
        self.optimizer.zero_grad()
        out = self.model(x).flatten()
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()
        normalized_out = torch.sigmoid(out)
        num_correct = torch.sum((out > 0) == (y == 1))
        TP = torch.sum((out > 0) * (y == 1))
        TN = torch.sum((out <= 0) * (y == 0))
        FP = torch.sum((out > 0) * (y == 0))
        FN = torch.sum((out <= 0) * (y == 1))
        return BatchResult(loss.item(), num_correct.item(),TP,TN,FP,FN, out, y)      

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        # x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        x = x.to(self.device, dtype=torch.float)        
        y = y.to(self.device, dtype=torch.float)

        with torch.no_grad():
            out = self.model(x).flatten()
            loss = self.loss_fn(out, y)
            num_correct = torch.sum((out > 0) == (y == 1))
            out_norm=torch.softmax(out,dim=-1)
            if self.classification_threshold==None:
                TP = torch.sum((out > 0) * (y == 1))
                TN = torch.sum((out <= 0) * (y == 0))
                FP = torch.sum((out > 0) * (y == 0))
                FN = torch.sum((out <= 0) * (y == 1))
            else:
                TP = torch.sum((out_norm > self.classification_threshold) * (y == 1))
                TN = torch.sum((out_norm <= self.classification_threshold) * (y == 0))
                FP = torch.sum((out_norm > self.classification_threshold) * (y == 0))
                FN = torch.sum((out_norm <= self.classification_threshold) * (y == 1))                
        return BatchResult(loss.item(), num_correct.item(),TP,TN,FP,FN, out, y)


class EcgImageToDigitizedTrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        y = (y[0].to(self.device, dtype=torch.float), y[1].to(self.device, dtype=torch.float))
        batch_size = y[0].shape[0]
        dim_ratio = y[0].nelement()/y[1].nelement()

        self.optimizer.zero_grad()

        out = self.model(x)
        loss = dim_ratio*self.loss_fn(out[0], y[0]) + self.loss_fn(out[1], y[1])
        loss.backward()
        self.optimizer.step()

        num_correct = batch_size*(torch.sum(torch.abs(out[0]-y[0]) < 0.01) + torch.sum(torch.abs(out[1]-y[1]) < 0.01))\
            / (torch.numel(y[0]) + torch.numel(y[1]))

        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.transpose(1, 2).transpose(1, 3).to(self.device, dtype=torch.float)
        y = (y[0].to(self.device, dtype=torch.float), y[1].to(self.device, dtype=torch.float))
        batch_size = y[0].shape[0]

        with torch.no_grad():
            out = self.model(x)
            loss = self.loss_fn(out[0], y[0]) + self.loss_fn(out[1], y[1])
            num_correct = \
                batch_size * (
                        torch.sum(torch.abs(out[0] - y[0]) < 0.01) + torch.sum(torch.abs(out[0] - y[0]) < 0.01)) \
                / (torch.numel(y[0]) + torch.numel(y[1]))

        return BatchResult(loss.item(), num_correct.item())


class Ecg12LeadImageNetAdversarialNet(Trainer):
    def train_batch(self, batch_source, batch_target, grl_lambda, batch_num = None, epoch_num = None) -> BatchResult:
        # print(f'Epoch num in train batch function : {epoch_num}')
        # torch.autograd.set_detect_anomaly(True) 
        self.grl_lambda = grl_lambda
        self.optimizer.zero_grad()
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr']/self.lr_factor_between_heads      
        x_t, y_t = batch_target
        x_t = x_t.to(self.device, dtype=torch.float)
        y_t = y_t.to(self.device, dtype=torch.float)      
        x_t[:,0,:,:]= (x_t[:,0,:,:]-torch.min(x_t[:,0,:,:]))/(torch.max(x_t[:,0,:,:])-torch.min(x_t[:,0,:,:]))
        x_t[:,1,:,:]= (x_t[:,1,:,:]-torch.min(x_t[:,1,:,:]))/(torch.max(x_t[:,1,:,:])-torch.min(x_t[:,1,:,:]))
        x_t[:,2,:,:]= (x_t[:,2,:,:]-torch.min(x_t[:,2,:,:]))/(torch.max(x_t[:,2,:,:])-torch.min(x_t[:,2,:,:]))
        # self.draw_batch(x_t,y_t, condition = 1.0)
        if epoch_num < self.epoch_to_start_disturbing:
            with torch.no_grad():             
                class_pred_t,  domain_pred_t= self.model(x_t)
                lt = self.loss_fn_domain(torch.squeeze(domain_pred_t),y_t).to(self.device, dtype=torch.float)    
        else:            
            for p in self.model.class_classifier.parameters(): p.requires_grad = False
            for p in self.model.domain_classifier.parameters(): p.requires_grad = True             
            class_pred_t,  domain_pred_t= self.model(x_t)
            lt = self.loss_fn_domain(torch.squeeze(domain_pred_t),y_t).to(self.device, dtype=torch.float)    
            lt.backward()
            del x_t
            self.optimizer.step()
            
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr']*self.lr_factor_between_heads       
        # Propagate source first
        x_s, y_s = batch_source
        # self.draw_batch(x_s,y_s)
        x_s = x_s.to(self.device, dtype=torch.float)
        y_s = y_s.to(self.device, dtype=torch.float)
        x_s[:,0,:,:]= (x_s[:,0,:,:]-torch.min(x_s[:,0,:,:]))/(torch.max(x_s[:,0,:,:])-torch.min(x_s[:,0,:,:]))
        x_s[:,1,:,:]= (x_s[:,1,:,:]-torch.min(x_s[:,1,:,:]))/(torch.max(x_s[:,1,:,:])-torch.min(x_s[:,1,:,:]))
        x_s[:,2,:,:]= (x_s[:,2,:,:]-torch.min(x_s[:,2,:,:]))/(torch.max(x_s[:,2,:,:])-torch.min(x_s[:,2,:,:]))        
        self.model.grl_lambda = grl_lambda
        for p in self.model.class_classifier.parameters(): p.requires_grad = True
        for p in self.model.domain_classifier.parameters(): p.requires_grad = False         
        class_pred_s,  domain_pred_s= self.model(x_s)
        ls = self.loss_fn_source(class_pred_s,torch.unsqueeze(y_s,-1) ).to(self.device, dtype=torch.float)           
        ls.backward()
        self.optimizer.step()                     

        ls_= ls.data.clone()
        lt_= lt.data.clone()
        out_s =torch.squeeze(class_pred_s)
        num_correct_s = torch.sum(  torch.squeeze(out_s > 0) ==  torch.squeeze(y_s == 1.) )
        TP_s = torch.sum( torch.squeeze(out_s > 0) *  torch.squeeze(y_s == 1.))
        TN_s = torch.sum( torch.squeeze(out_s <= 0) *  torch.squeeze(y_s == 0))
        FP_s = torch.sum( torch.squeeze(out_s > 0) *  torch.squeeze(y_s == 0))
        FN_s = torch.sum( torch.squeeze(out_s <= 0) *  torch.squeeze(y_s == 1.))
        
        out= torch.squeeze(domain_pred_t).clone()
        out_t =torch.squeeze(out).clone()
        num_correct_t = torch.sum(  torch.squeeze(out_t > 0) == torch.squeeze(y_t == 1.) )
        TP_t = torch.sum( torch.squeeze(out_t > 0) *  torch.squeeze(y_t == 1.))
        TN_t = torch.sum( torch.squeeze(out_t <= 0) *  torch.squeeze(y_t == 0))
        FP_t = torch.sum( torch.squeeze(out_t > 0) *  torch.squeeze(y_t == 0))
        FN_t = torch.sum( torch.squeeze(out_t <= 0) *  torch.squeeze(y_t == 1.))
        #ls,lt
        return BatchResult_complex_net(ls_,lt_, num_correct_s.item(),
            num_correct_t.item(),TP_s,TN_s,FP_s,FN_s,TP_t,TN_t,FP_t,FN_t,
            out_s, y_s, out_t, y_t)      

    def test_batch(self, batch_source, batch_target, grl_lambda = 0, epoch_num = None) -> BatchResult:
        x_s, y_s = batch_source
        # self.draw_batch(x_s,y_s)
        x_t, y_t = batch_target
        # self.draw_batch(x_t,y_t, condition = 1.0)
        x_s = x_s.to(self.device, dtype=torch.float)
        y_s = y_s.to(self.device, dtype=torch.float)
        x_t = x_t.to(self.device, dtype=torch.float)
        y_t = y_t.to(self.device, dtype=torch.float)    
        x_t[:,0,:,:]= (x_t[:,0,:,:]-torch.min(x_t[:,0,:,:]))/(torch.max(x_t[:,0,:,:])-torch.min(x_t[:,0,:,:]))
        x_t[:,1,:,:]= (x_t[:,1,:,:]-torch.min(x_t[:,1,:,:]))/(torch.max(x_t[:,1,:,:])-torch.min(x_t[:,1,:,:]))
        x_t[:,2,:,:]= (x_t[:,2,:,:]-torch.min(x_t[:,2,:,:]))/(torch.max(x_t[:,2,:,:])-torch.min(x_t[:,2,:,:]))
        x_s[:,0,:,:]= (x_s[:,0,:,:]-torch.min(x_s[:,0,:,:]))/(torch.max(x_s[:,0,:,:])-torch.min(x_s[:,0,:,:]))
        x_s[:,1,:,:]= (x_s[:,1,:,:]-torch.min(x_s[:,1,:,:]))/(torch.max(x_s[:,1,:,:])-torch.min(x_s[:,1,:,:]))
        x_s[:,2,:,:]= (x_s[:,2,:,:]-torch.min(x_s[:,2,:,:]))/(torch.max(x_s[:,2,:,:])-torch.min(x_s[:,2,:,:]))           
        with torch.no_grad():
            out_source, _ = self.model(x_s)
            _ , out_domain = self.model(x_t)
            out_source = torch.squeeze(out_source)
            out_domain = torch.squeeze(out_domain)
            loss_source = self.loss_fn(out_source, y_s)
            loss_domain = self.loss_fn(out_domain, y_t)
            y_s =torch.squeeze(y_s)
            y_t =torch.squeeze(y_t)
            num_correct_source = torch.sum((out_source > 0) == (y_s== 1.0))
            num_correct_domain = torch.sum((out_domain > 0) == (y_t== 1.0))

            if self.classification_threshold==None:
                TP_s = torch.sum((out_source > 0) * (y_s == 1.0))
                TN_s = torch.sum((out_source <= 0) * (y_s == 0.0))
                FP_s = torch.sum((out_source > 0) * (y_s == 0.0))
                FN_s = torch.sum((out_source <= 0) * (y_s == 1.0))
                TP_d = torch.sum((out_domain > 0) * (y_t == 1.0))
                TN_d = torch.sum((out_domain <= 0) * (y_t == 0.0))
                FP_d = torch.sum((out_domain > 0) * (y_t == 0.0))
                FN_d = torch.sum((out_domain <= 0) * (y_t == 1.0))                
            else:
                TP_s = torch.sum((out_source > self.classification_threshold) * (y_s == 1.0))
                TN_s = torch.sum((out_source <= self.classification_threshold) * (y_s == 0.0))
                FP_s = torch.sum((out_source > self.classification_threshold) * (y_s == 0.0))
                FN_s = torch.sum((out_source <= self.classification_threshold) * (y_s == 1.0))          
                TP_d = torch.sum((out_domain > self.classification_threshold) * (y_t == 1.0))
                TN_d = torch.sum((out_domain <= self.classification_threshold) * (y_t == 0.0))
                FP_d = torch.sum((out_domain > self.classification_threshold) * (y_t == 0.0))
                FN_d = torch.sum((out_domain <= self.classification_threshold) * (y_t == 1.0))                

        return BatchResult_complex_net(loss_source.item(),loss_domain.item(),
            num_correct_source.item(),num_correct_domain.item(),TP_s,TN_s,FP_s,
            FN_s,TP_d,TN_d,FP_d,FN_d, out_source,y_s,out_domain,y_t)
    

    def draw_batch(self, batch_to_draw, labels, condition = None):
        for image_num in range(len(labels)):
            if (condition is None) or (condition  == labels[image_num].item()):
                img = batch_to_draw[image_num]
                img = np.transpose(img.cpu().detach().numpy(),(1,2,0))
                plt.imshow(img)
                plt.title(str(labels[image_num].item()))
                plt.show()



