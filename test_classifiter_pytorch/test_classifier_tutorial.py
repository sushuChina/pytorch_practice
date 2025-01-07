#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:45:37 2024

@author: sushu
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np
import matplotlib.pyplot as plt

from models.network_class_filter import LeNetRGB


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_and_shaow_data():   
    my_data = unpickle('/home/sushu/code/python/pytorch_tutorial/datasets/cifar-10-batches-py/data_batch_1')
    
    img_data = my_data[b'data']
    img_label = my_data[b'labels']
    
    img_1 = img_data[6,:]
    
    img_reshape = np.reshape(img_1,(3,32,32))
    img_color = img_reshape.transpose((1,2,0))
    
    plt.imshow(img_color)

if __name__=='__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    
    #%% Load and normalize CIFAR10
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #%% define a convolutional Neural network
    net = LeNetRGB()
    net.to(device)
    net.load_state_dict(torch.load('./datasets/weights/cifar_net.pth'))
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))