#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:28:01 2024

@author: sushu
"""

import torch
import torch.nn as nn
import torch.nn.functional as nf

class MyNN(nn.Module):
    def __init__(self, in_size=256, layer_num=100):
        super(MyNN, self).__init__()
        self.in_size = in_size
        # layer_num层全连接堆叠起来
        self.FC = nn.Sequential(*[nn.Linear(in_size, in_size, bias=False) for _ in range(layer_num)])
        # 不同初始化方法测试
        self._initialize()
    
    # def print_(self):
        
        
    def forward(self, x):
        for (idx, linear) in enumerate(self.FC):
            x = linear(x)
            # 打印每一层的均值和标准差
            print("idx:{}, mean:{}, std:{}".format(idx + 1, x.mean(), x.std()))
            if torch.isnan(x.mean()) or torch.isnan(x.std()) or torch.isinf(x.mean()) or torch.isinf(x.std()):
                break
        return x
    
    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 固定值
                nn.init.constant(m.weight.data, 1)
                
if __name__ == '__main__':
    mynn = MyNN(in_size=256, layer_num=20)
    
    x = torch.randn(8, 256)
    x.requires_grad = True
    x.retain_grad()
    y = mynn(x)
    
    y.backward(torch.ones_like(y))
    print('x grad mean:', x.grad.mean())
    print('x grad std:', x.grad.std())