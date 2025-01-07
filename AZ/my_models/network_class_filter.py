#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:02:09 2024

@author: sushu
"""


import torch
import torch.nn as nn
import torch.nn.functional as nnf

class LeNetRGB(nn.Module):
    def __init__(self):
        super(LeNetGray,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(nnf.relu(self.conv1(x)))
        x = self.pool(nnf.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nnf.relu(self.fc1(x))
        x = nnf.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNetGray(nn.Module):
    def __init__(self):
        super(LeNetGray,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool(nnf.relu(x))
        x = self.pool(nnf.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nnf.relu(self.fc1(x))
        x = nnf.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TestNet(nn.Module):
    
    def __init__(self):
        super(TestNet,self).__init__()        
        self.conv1 = nn.Conv2d(1, 6, 5)    # C1 (Convolution) <1*32*32> to <6*28*28> (no padding)
        self.conv2 = nn.Conv2d(6, 16, 5)   # C3 (Convolution) <6*14*14> to <16*10*10> (no padding)
        
        self.fc1 = nn.Linear(16*5*5, 120)  # F5 (full connection1) <16*5*5> to <120*1>  (input must be flattened)
        self.fc2 = nn.Linear(120, 84)      # F6 (full connection1) <120*1> to <84*1>
        self.fc3 = nn.Linear(84, 10)       # OUTPUT (full connection1) <84*1> to <10*1> 
        
    def forward(self,input):
        # Convolution layer---c1
        # input shape:N*1*32*32
        # output shape:N*6*28*28
        c1 = nnf.relu(self.conv1(input))
        # Subsampling---s2
        # input shape:N*6*28*28
        # output shape:N*6*14*14
        s2 = nnf.max_pool2d(c1,(2,2))
        # Convolution layer---c3
        # input shape:6*14*14
        # output shape:6*10*10
        c3 = nnf.relu(self.conv2(s2))
        # Subsampling---s4
        # input shape:N*6*10*10
        # output shape:N*16*5*5
        s4 = nnf.max_pool2d(c3,(2,2))
        # flattened
        # input shape:N*16*5*5
        # output shape:N*1*400
        s4 = torch.flatten(s4,1)
        # full connection1 F5
        # input shape:N*1*400
        # output shape:N*1*120
        f5 = nnf.relu(self.fc1(s4))
        # full connection1 F6
        # input shape:N*1*120
        # output shape:N*1*84
        f6 = nnf.relu(self.fc2(f5))
        # full connection1 F7
        # input shape:N*1*84
        # output shape:N*1*10
        output = self.fc3(f6)
        return output
