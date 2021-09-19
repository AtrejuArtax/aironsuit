import os
import glob
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch


def fit(module, x_train, y_train, x_val=None, y_val=None, best_module_name=None, **kwargs):

    # Invoke trainer
    kwargs_ = kwargs.copy()
    if 'epochs' in kwargs_.keys():
        kwargs_['max_epochs'] = kwargs_['epochs']
        del kwargs_['epochs']
    trainer = Trainer(**kwargs_)

    # Converting numpy arrays to Tensor datasets
    device = set_up_cuda()[0]
    training_dataset = array_to_tensor_dataset(x_train, y_train, device=device)
    if x_val and y_val:
        val_dataset = array_to_tensor_dataset(x_val, y_val, device=device)
    else:
        val_dataset = None

    # Train module
    if val_dataset:
        trainer.fit(module, DataLoader(training_dataset), DataLoader(val_dataset))
    else:
        trainer.fit(module, DataLoader(training_dataset))


def evaluate(module, x_val, y_val):
    trainer = Trainer()
    device = set_up_cuda()[0]
    val_dataset = array_to_tensor_dataset(x_val, y_val, device=device)
    return trainer.validate(module, DataLoader(val_dataset))


def set_up_cuda():
    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    return cuda, device


def array_to_tensor_dataset(x, y, device=None):
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    if device:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)
    return TensorDataset(x_tensor, y_tensor)
