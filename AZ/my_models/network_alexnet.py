#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:11:08 2024

@author: sushu
"""


import torch
import torch.nn as nn
import torch.nn.functional as nnF


#%% input-size=227
class my_AlexNet(nn.Module):
    def __init__(self):
        super(my_AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5,padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3,padding=1)
        self.conv4 = nn.Conv2d(256, 384, 3,padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3,padding=1)
        self.fc1 = nn.Linear(43264, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.LRN = nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2)
        self.Dropout = nn.Dropout(p=0.5, inplace=True)
    
    def forward(self,input):
        c1 = nnF.max_pool2d(self.LRN(nnF.relu(self.conv1(input))),kernel_size=3,stride=2)
        c2 = nnF.max_pool2d(self.LRN(nnF.relu(self.conv2(c1))),kernel_size=3,stride=2)
        c3 = nnF.relu(self.conv3(c2))
        c4 = nnF.relu(self.conv4(c3))
        c5 = nnF.max_pool2d(nnF.relu(self.conv5(c4),kernel_size=3,stride=2))
        s1 = torch.flatten(c5,1)
        fc1 = nnF.relu(self.fc1(self.Dropout(s1)))
        fc2 = nnF.relu(self.fc2(self.Dropout(fc1)))
        fc3 = self.fc3(fc2)
        
        return fc3
    
#%% input-size=227
class my_AlexNet_Nclass1(nn.Module):
    def __init__(self,num_classes = 2):
        super(my_AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5)
        self.conv3 = nn.Conv2d(256, 384, 3)
        self.conv4 = nn.Conv2d(256, 384, 3)
        self.conv5 = nn.Conv2d(384, 256, 3)
        self.fc1 = nn.Linear(9216, 500)
        self.fc2 = nn.Linear(500, 20)
        self.fc3 = nn.Linear(20, num_classes)
        self.LRN = nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2)
        self.Dropout = nn.Dropout(p=0.5, inplace=True)
        
    def forward(self,input):
        c1 = nnF.max_pool2d(self.LRN(nnF.relu(self.conv1(input))),kernel_size=3,stride=2)
        c2 = nnF.max_pool2d(self.LRN(nnF.relu(self.conv2(c1))),kernel_size=3,stride=2)
        c3 = nnF.relu(self.conv3(c2))
        c4 = nnF.relu(self.conv4(c3))
        c5 = nnF.max_pool2d(nnF.relu(self.conv5(c4),kernel_size=3,stride=2))
        s1 = torch.flatten(c5,1)
        fc1 = nnF.relu(self.fc1(self.Dropout(s1)))
        fc2 = nnF.relu(self.fc2(self.Dropout(fc1)))
        fc3 = self.fc3(fc2)
        
        return fc3
        
#%% input-size=227
class my_AlexNet_Nclass2(nn.Module):
    def __init__(self,num_classes = 2):
        super().__init__()
        self.backbone_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            )
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features = (256*6*6), out_features = 500),
            nnF.relu(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features = 500, out_features = 20),
            nnF.relu(),
            nn.Linear(in_features = 20, out_features = num_classes)
            )
    def forward(self,input):
        X = self.backbone_net(input)
        X = X.veiw(-1,256*6*6)#same as X = torch.flatten(X)
        return self.classifier_head(X)
    
    
    