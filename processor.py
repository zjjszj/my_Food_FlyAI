# -*- coding: utf-8 -*
import cv2
import numpy
import numpy as np
from flyai.processor.base import Base
from flyai.processor.download import check_download
import torch
from path import DATA_PATH
#my
from PIL import Image
from torchvision import transforms as T

class Processor(Base):

    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH) #DATA_PATH=/data/train_code/tain63cd6642bd2d8d24de80cac9/data/input
        x = Image.open(path).convert('RGB')
        w,h=x.size
        if h<512:
            x = T.Resize((512,int(512 * (512 / h))))(x)
        if w<512:
            x=T.Resize((int(512*(512/w)),512))(x)
        x=T.RandomCrop((448,448))(x)
        x=T.RandomHorizontalFlip()(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return x

    def input_y(self, labels):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        labels:0/1/2...
        '''
        return torch.tensor([labels])

    def output_x(self, image_path):
        '''
        测试时使用，数据增强
        '''
        path = check_download(image_path, DATA_PATH)
        x = Image.open(path).convert('RGB')
        w,h=x.size
        if h<512:
            x = T.Resize((512,int(512 * (512 / h))))(x)
        if w<512:
            x=T.Resize((int(512*(512/w)),512))(x)
        x=T.CenterCrop((448,448))(x)
        #x=T.RandomCrop((448,448))(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return x

    def output_y(self, data):
        '''
        测试时使用，把模型输出的y转为对应的结果
        '''
        output = np.argmax(data)
        return output
