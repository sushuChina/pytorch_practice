#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:06:48 2024

@author: sushu
"""
from torch.utils.data import DataLoader

from .MNIST import MNIST_NPY

dataset_hub = {'MNIST_NPY':MNIST_NPY}


def get_dataset(config):
    if config.dataset in dataset_hub.keys():
        dataset = dataset_hub[config.dataset](config,config.mode)
    else:
        raise NotImplementedError('Unsupported dataset!')
        
    return dataset
    
    
def get_loader(config):
    train_dataset = get_dataset(config)
    
    # Make sure train number is divisible by train batch size
    config.train_num = int(len(train_dataset) // config.train_bs * config.train_bs)

    data_loader = DataLoader(train_dataset, batch_size=config.train_bs, 
                                shuffle=True, num_workers=config.num_workers, drop_last=True)

    return data_loader


# def get_test_loader(config): 
#     from .test_dataset import TestDataset
#     dataset = TestDataset(config)

#     config.test_num = len(dataset)

#     if config.DDP:
#         raise NotImplementedError()

#     else:
#         test_loader = DataLoader(dataset, batch_size=config.test_bs, 
#                                     shuffle=False, num_workers=config.num_workers)

#     return test_loader