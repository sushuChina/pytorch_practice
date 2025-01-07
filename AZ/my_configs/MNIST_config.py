#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:45:10 2025

@author: sushu
"""
from .base_config import BaseConfig

class MNIST_NPY_CONFIG():
    def __init__(self,mode = 'train'):
        
        # data set 
        self.dataset = 'MNIST_NPY'
        self.num_class = 10
        self.mode = mode
        self.num_workers = 2
        if mode == 'train':
            self.data_path = '/home/sushu/code/python/pytorch_tutorial/datas/MNIST/train/MNIST_train_data.npy'
            self.label_path = '/home/sushu/code/python/pytorch_tutorial/datas/MNIST/train/MNIST_train_labels.npy'
        elif mode == 'test':
            self.data_path = '/home/sushu/code/python/pytorch_tutorial/datas/MNIST/test/MNIST_test_data.npy'
            self.label_path = '/home/sushu/code/python/pytorch_tutorial/datas/MNIST/test/MNIST_test_labels.npy'
            
        
        # train
        self.train_bs = 64
        self.optimizer_type = 'adam'
        self.lr = 0.001
        self.momentum = 0.9
        
        # DDP
        self.DDP = False