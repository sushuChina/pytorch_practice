#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:45:37 2024

@author: sushu

Tutorial URL:https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__=='__main__':
    #%% Load and normalize CIFAR10
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    learning_rate = 0.001
    momentum_ = 0.9
    
    batch_size = 4
    max_epoch = 6
    
    
    trainset = torchvision.datasets.CIFAR10(root='../datas', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='../datas', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #%% show some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    
    #%% define a convolutional Neural network
    net = Net()
    
    # Define a Loss function and optimizer
    loss_func = nn.CrossEntropyLoss()    
    #定义优化器
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,momentum = momentum_)
    
    #%% Train the network 
    for epoch in range(max_epoch):
        
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels = data
            #初始化梯度
            optimizer.zero_grad()
            #保存训练结果
            outputs = net(inputs)
            #计算损失和
            #多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题, 通常使用sigmod
            loss = loss_func(outputs,labels)
            #反向传播            
            loss.backward()
            #更新参数
            optimizer.step()
        
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
               print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
               running_loss = 0.0
               
    print('Finished Training')
    PATH = './weights/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))