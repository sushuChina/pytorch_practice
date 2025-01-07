#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:50:15 2024

@author: sushu
"""
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchviz import make_dot

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)


prediction = model(data) # forward pass
p_np = prediction.detach().numpy()

# loss function
loss = (prediction - labels).sum()
loss.backward() # backward pass

# optimizer+
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# initiate gradient descent.
optim.step() #gradient descent

prediction2 = model(data)
p2_np = prediction2.detach().numpy()


#%% Autograd 显示传递

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = (2*a**3-b**2)

# # 显式传递----sum Error
# external_grad = torch.tensor([1., 1.])
# Q.backward(gradient=external_grad)

# 隐式传递---Mean Squared Error, MSE
Q_MSE = torch.mean(Q**2)
Q_MSE.backward()
