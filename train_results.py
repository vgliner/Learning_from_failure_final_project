from typing import NamedTuple, List
from torch.utils.data import DataLoader



class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_correct: int
    num_TP: int
    num_TN: int
    num_FP: int
    num_FN: int
    out : float
    y : int

class BatchResult_complex_net(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss_s: float
    loss_t: float
    num_correct_source: int
    num_correct_target: int
    num_TP_source: int
    num_TN_source: int
    num_FP_source: int
    num_FN_source: int
    num_TP_target: int
    num_TN_target: int
    num_FP_target: int
    num_FN_target: int    
    out_source : float
    y_source : int
    out_target : float
    y_target : int
    

class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float
    num_TP: int
    num_TN: int
    num_FP: int
    num_FN: int
    out : float
    y : int 
    ROC_AUC : float

class EpochResult_complex_net(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses_s: List[float]
    losses_t: List[float]
    accuracy_source: float
    accuracy_target: float
    num_TP_source: int
    num_TN_source: int
    num_FP_source: int
    num_FN_source: int
    num_TP_target: int
    num_TN_target: int
    num_FP_target: int
    num_FN_target: int   
    out_source : float
    y_source : int
    out_target : float
    y_target : int
    grl_lambda : float

class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]
    TPs: List[int]
    TNs: List[int]
    FPs: List[int]
    FNs: List[int] 
    Is_checkpointed: List[bool] 
    ROC_AUCs: List[float]  

class FitResult_complex_net(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc_source: List[float]
    train_acc_target: List[float]
    test_loss_source: List[float]
    test_acc_source: List[float]
    test_loss_target: List[float]
    test_acc_target: List[float]    
    TPs_source: List[int]
    TNs_source: List[int]
    FPs_source: List[int]
    FNs_source: List[int] 
    TPs_target: List[int]
    TNs_target: List[int]
    FPs_target: List[int]
    FNs_target: List[int]     
    Is_checkpointed: List[bool] 
    ROC_AUCs: List[float]  
    ROC_AUCt: List[float]      