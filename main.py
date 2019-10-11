# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import torch
from flyai.dataset import Dataset
from model import Model
from net import Net
from path import MODEL_PATH
import time
from torch import nn
import numpy as np

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

'''
实现自己的网络结构
'''
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
net = Net().to(device)
net = net.cuda()

'''
dataset.get_step() 获取数据的总批次

'''

def optim_policy(model):
    # 返回第一层和全连阶层的权重
    needed_optim = []
    for param in model.features[0].parameters():
        needed_optim.append(param)
    for param in model.classifier.parameters():
        needed_optim.append(param)
    return needed_optim

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.model.parameters()), lr=0.0001)
# 损失函数
cross_criterion = nn.CrossEntropyLoss()

print('dataset.get_step()======', dataset.get_step())
epoch_steps = dataset.get_step() // args.EPOCHS
def adjust_lr(optimizer, step,batch_size):
    if batch_size==32:
        if step < 5*epoch_steps:
            lr = 1e-4
        else:
            lr = 1e-5*3
        for p in optimizer.param_groups:
            p['lr'] = lr
    else:

        if step < 150:
            lr = 1e-4 * (step // 120 + 1)
        elif step < 360:
            lr = 1e-4
        else:
            lr = 1e-5
        for p in optimizer.param_groups:
            p['lr'] = lr

val_step = 1  # 单位：epoch
correct = 0
best_acc = 0

for step in range(dataset.get_step()):
    #adjust_lr(optimizer, step,args.BATCH)
    best_score = 0
    x_train, y_train = dataset.next_train_batch()  ## type:numpy.ndarray
    x_train = torch.tensor(x_train).cuda()
    y_train = torch.tensor(y_train.reshape(-1)).cuda()
    #print('实际标签为：',y_train)
    output = net(x_train)
    #pre = output.max(1)[1]
    #print('预测的标签为：',pre)
    loss = cross_criterion(output, y_train)
    print('loss_per_step==', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # acc
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(y_train.data.view_as(pred)).cpu().sum()
    if (step + 1) % epoch_steps == 0:
        print('lr==',optimizer.param_groups[0]['lr'])
        print('train_acc_per_epoch\tepoch={}\t[{}]/[{}]\ttrain_acc={}'.format((step + 1) // epoch_steps,
                        correct, epoch_steps * args.BATCH,100 * correct.numpy() / (epoch_steps * args.BATCH)))
        print('train_loss_per_epoch\tepoch={}\tloss={}'.format((step + 1) // epoch_steps, loss.item()))
        correct = 0
    # val
    if (step + 1) % (val_step * epoch_steps) == 0:
        correct = 0
        # val data:32
        x_val, y_val = dataset.next_validation_batch()
        #print('验证的实际标签：',y_val)
        x_val = torch.tensor(x_val).cuda()
        y_val = torch.tensor(y_val.reshape(-1)).cuda()
        output = net(x_val)
        #print('验证预测的标签：',output.max(1)[1])
        loss = cross_criterion(output, y_val)
        print('lr==',optimizer.param_groups[0]['lr'])
        print('val_loss==', loss.item())
        # acc
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y_val.data.view_as(pred)).cpu().sum()
        print('val_acc\tepoch={}\t[{}]/[{}]\t{}'.format((step + 1) // epoch_steps,
                                            correct, len(y_val), 100 * correct.numpy() / len(y_val)))
        if (correct.numpy() / len(y_val))>best_acc:
            best_acc=correct.numpy() / len(y_val)
            #保存模型
            model.save_model(net.model, MODEL_PATH, overwrite=True)
            print('===saved successful!===')
        correct=0
