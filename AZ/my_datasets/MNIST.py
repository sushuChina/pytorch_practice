#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:22:58 2025

@author: sushu


"""
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MNIST_NPY(Dataset):
    def __init__(self,config,mode = 'train'):
        super().__init__()
        self.datas = np.load(config.data_path)
        self.labels = np.load(config.label_path)
        #%% define transforms
        if mode == 'train':
            self.transforms = transforms.Compose([        #随机旋转图片
                transforms.RandomHorizontalFlip(),
                #将图片尺寸resize到32x32
                transforms.Resize((32,32)),
                #将图片转化为Tensor格式
                transforms.ToTensor(),
                #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
                transforms.Normalize((0.1307,), (0.3081,))

                ])
        elif mode == 'test':
             self.transforms = transforms.Compose([
                #将图片尺寸resize到32x32
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
                ])
        else:
            raise ValueError("mode should be 'train' or 'test'")
            
    def __getitem__(self,index):
        img_array = np.reshape(self.datas[index,:],(28,28))
        image = Image.fromarray(img_array)

        label = self.labels[index]
        image = self.transforms(image)
        return image,label
        
    def __len__(self):
        return self.datas.shape[0]