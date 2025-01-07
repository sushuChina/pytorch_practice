#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:06:16 2024

@author: sushu
"""

import torch
import torch.nn as Tnn
import torch.nn.functional as Tnf
import torch.optim as optim

from models.network_class_filter import TestNet

if __name__ == "__main__":

    net = TestNet()
    # print(net)

    # params = list(net.parameters()) # learnable parameters
    # print(len(params))
    # print(params[0].size())  # conv1's .weight
    
    input_data = torch.randn(1, 1, 32, 32)
    # out = net(input_data)
    # print(out)
    
    # # Zero the gradient buffers of all parameters and backprops with random gradients:
    # net.zero_grad()
    # out.backward(torch.randn(1, 10))
    
    output = net(input_data)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    
    criterion = Tnn.MSELoss()
    # loss = criterion(output, target)
    # print(loss)
    
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)    
        
    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input_data)
    loss = criterion(output, target)
    loss.backward()
    # optimizer.step()    # Does the update

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    