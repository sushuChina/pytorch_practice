#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 09:16:23 2025

@author: sushu
"""

import torch
import torch.nn.functional as nnf
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from AZ import my_util,my_datasets
from AZ.my_models import LeNetGray
import AZ.my_configs as my_configs

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pre_trans = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    image_name = '9_4.png'    
    test_image = cv2.imread(image_name)
    # pre process image
    img_array = 255-cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)#图片转为灰度图，因为mnist数据集都是灰度图 
    img_array = cv2.resize(img_array,dsize=(32,32),interpolation=cv2.INTER_NEAREST)
    img_norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    img = Image.fromarray(img_norm)
    img = pre_trans(img)
    img_numpy = img.detach().numpy()
    img_numpy = img_numpy[0]
    img = img.to(device)
    img = img.unsqueeze(0)
    
    # MNIST
    index = 600
    MNIST_datas = np.load('/home/sushu/code/python/pytorch_tutorial/datas/MNIST/train/MNIST_train_data.npy')
    MNIST_image_1 = np.reshape(MNIST_datas[index],(28,28))
    img_MNIST = Image.fromarray(MNIST_image_1)
    img_MNIST_ = pre_trans(img_MNIST)
    img_MNIST_numpy = img_MNIST_.detach().numpy()
    img_MNIST_ = img_MNIST_.unsqueeze(0)
    img_MNIST_ = img_MNIST_.to(device)
    img_MNIST_numpy = img_MNIST_numpy[0]
    MNIST_labels = np.load('/home/sushu/code/python/pytorch_tutorial/datas/MNIST/train/MNIST_train_labels.npy')
    MNIST_labels = MNIST_labels[index]
    
    # load model paramter
    # model = LeNetGray()
    # state = torch.load("../datas/weights/model-mnist-weight.pth", map_location = 'cpu', weights_only=False)
    # model.load_state_dict(state)
    # load model
    model  = torch.load("../datas/weights/model-mnist.pth", map_location = 'cpu',weights_only=False)
    
    model.to(device)
    model.eval()
    
    output = model(img)
    label = output.argmax(dim=1)
    output_array = label.detach().numpy()
    
    

    
    