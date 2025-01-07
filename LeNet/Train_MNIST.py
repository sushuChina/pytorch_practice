#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:53:25 2024

@author: sushu

@article{lecun2010mnist,
         title={MNIST handwritten digit database},
         author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
         journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
         volume={2},
         year={2010}
}

"""

import torch
import numpy as np
from torchvision import datasets
from torch.optim import SGD, Adam, AdamW
import torch.nn.functional as nnf
import time

from AZ import my_util,my_datasets
from AZ.my_models import LeNetGray
import AZ.my_configs as my_configs 




def get_optimizer(config, model):    
    optimizer_hub = {'sgd':SGD, 'adam':Adam, 'adamw':AdamW}
    params = model.parameters()

    if config.optimizer_type == 'sgd':
        # config.lr = config.base_lr * config.gpu_num
        optimizer = optimizer_hub[config.optimizer_type](params=params, lr=config.lr, 
                                                    momentum=config.momentum) 
                                                    # ,weight_decay=config.weight_decay)

    elif config.optimizer_type in ['adam', 'adamw']:
        # config.lr = 0.001 * config.gpu_num
        optimizer = optimizer_hub[config.optimizer_type](params=params, lr=config.lr)

    else:
        raise NotImplementedError(f'Unsupported optimizer type: {config.optimizer_type}')

    return optimizer

#%% download datas 
def download_data_from_pytorch():
    train_data = datasets.MNIST(root="./datas/", train=True, download=True)
    test_data = datasets.MNIST(root="./datas/", train=False, download=True)
    
    # load datasets
    train_data = my_util.load_mnist_data('./datas/MNIST/train/')
    test_data = my_util.load_mnist_data('./datas/MNIST/test/',kind='t10k')
    # save_npy_data
    np.save('./datas/MNIST/train/MNIST_train_data.npy',train_data[0])
    np.save('./datas/MNIST/train/MNIST_train_labels.npy',train_data[1])
    np.save('./datas/MNIST/test/MNIST_test_data.npy',test_data[0])
    np.save('./datas/MNIST/test/MNIST_test_labels.npy',test_data[1])
#%% train function
def train_main(model,config,device,trainloader,optimizer,epoch):    
    # model训练模型, 启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
    model.train() 

    total = 0
    correct = 0.0   
    
    for i,data in enumerate(trainloader,0):
        
        input_image,labels = data
        input_image = input_image.to(device)
        labels = labels.to(device)
        # Initializing the Optimizer
        optimizer.zero_grad()
        # forward
        output = model(input_image)
        output_lable = nnf.log_softmax(output,dim=1)
        # loss function
        loss = nnf.cross_entropy(output_lable, labels)
        # 
        predict = output_lable.argmax(dim=1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        # backpropagation
        loss.backward()
        # updata params
        optimizer.step()
        
        
        if i % 1000 == 0:
            #loss.item()表示当前loss的数值
            print("Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, loss.item(), 100*(correct/total)))
            # Loss.append(loss.item())
            # Accuracy.append(correct/total)
        
    return loss.item(), correct/total
#%% test function
def test_main(model,device,testloader):
    # model val模式
    model.eval()

    #统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for data,label in testloader:
            data,label =  data.to(device),label.to(device)
            output = model(data)
            output_lable = nnf.log_softmax(output,dim=1)
            test_loss += nnf.cross_entropy(output_lable, label).item()
            predict = output_lable.argmax(dim=1)
            #计算正确数量
            total += label.size(0)
            correct += (predict == label).sum().item()
        #计算正确率
        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/total, 100*(correct/total)))

    
#%% train
if __name__ == '__main__':

    Loss = []
    Accuracy = []
    
    # datas
    config = my_configs.MNIST_NPY_CONFIG()
    train_dataloder = my_datasets.get_loader(config)
    config.epoch = 10
    
    config_test = my_configs.MNIST_NPY_CONFIG(mode='test')
    test_dataloder = my_datasets.get_loader(config_test)
    
    # set model device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # models
    model = LeNetGray()
    model.to(device)
    # optimizer
    optimizer = get_optimizer(config,model)
    
    for epoch in range(1,config.epoch+1):
        print("start_time",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        loss,acc = train_main(model,config,device,train_dataloder,optimizer,epoch)
        Loss.append(loss)
        Accuracy.append(acc)
        test_main(model,device,test_dataloder)
        print("end_time",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    
    print(model)
    torch.save(model, '../datas/weights/model-mnist.pth') #保存模型
    torch.save(model.state_dict(), '../datas/weights/model-mnist-weight.pth') #保存模型权重
