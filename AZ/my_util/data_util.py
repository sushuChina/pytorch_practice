#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:22:20 2024

@author: sushu
"""

import pickle
import os
import struct
import numpy as np
    
def unpickle_data(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_mnist_data(path, kind='train'): 
    """
    path:数据集的路径
    kind:值为train，代表读取训练集
    """   
    labels_path = os.path.join(path,f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path,f'{kind}-images-idx3-ubyte')
    #不再用gzip打开文件
    with open(labels_path, 'rb') as lbpath:
	    #使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
	    #这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II',lbpath.read(8))
        #使用np.fromfile读取剩下的数据
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        # 使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        #这样读到的前两个数据分别是magic number和样本个数
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        #使用np.fromfile读取剩下的数据
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    # 用gzip打开文件示例
    # with gzip.open(images_path, 'rb') as imgpath:
    #     magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
    #     images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        
    return images, labels

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels