#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:01:44 2024

@author: sushu
"""
import random 
import numpy as np
import torch

# python random
r_python = random.random()

random.seed(0)
r_python_seed = random.random()

# numpy random
r_np = np.random.random([2,2])

np.random.seed(0)
r_np_seed = np.random.random([2,2])

# pytorch random
rt = torch.rand((2,2))
rt_np = rt.numpy()

torch.manual_seed(0)
rt_seed = torch.rand((2,2))
rt_seed_np = rt_seed.numpy()