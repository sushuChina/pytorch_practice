#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:47:28 2024

@author: sushu
"""

import torch
import numpy as np

# initialization
## from data
data = [[1,2,],[3,4]]
data_tensor = torch.tensor(data)

## from ndarray
data_np = np.array(data)
data_tensor_np = torch.from_numpy(data_np)

## from another tensor
x_ones = torch.ones_like(data_tensor_np)
x_rand = torch.rand_like(data_tensor_np,dtype=torch.float)

## from a shape
shape = (2,3)
rand_tensor = torch.rand(shape)
zeros_tensor = torch.zeros(shape)
ones_tensor = torch.ones(shape)

# Attributes
m_tensor = torch.ones((4,4))

## shape
print(f"m_tensor's shape is:{m_tensor.shape}")
print(f"m_tensor's data type is:{m_tensor.dtype}")
print(f"m_tensor's device is:{m_tensor.device}") # the device on which they are stored---cpu or gpu

# Operations

## change device
if torch.cuda.is_available():
  m_tensor = m_tensor.to('cuda')
  print(f"Device tensor is stored on: {m_tensor.device}")
  
## cat 
t1 = torch.cat([m_tensor, m_tensor, m_tensor], dim=1)
print(t1)

## mul
m_tensor[:,1] = 0
print(f"tensor.mul(tensor) \n {m_tensor.mul(m_tensor)} \n")
print(f"tensor*tensor\n {m_tensor*m_tensor} \n")

## matmul
m_tensor[:,1] = 0
print(f"tensor.matmul(tensor) \n {m_tensor.matmul(m_tensor.T)} \n")
print(f"tensor@tensor \n {m_tensor@m_tensor.T} \n")

# bridge with numpy

## tensor 2 numpy
m_np = m_tensor.numpy()
print(m_np)
