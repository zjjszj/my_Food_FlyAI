# -*- coding: utf-8 -*
from torch import nn
import torchvision
import torch
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.model=self.model_process()
        model=models.resnet50(True)
        self.model=self.get_resnet50(model)

    def forward(self, x):
        return self.model(x)

    #vgg16_bn
    def model_process(self):
        # 加载模型
        vgg16_bn = torchvision.models.vgg16_bn(True)
        # print(vgg16_bn.classifier[6].out_features)      #1000
        # 修改全连接层最后一层的out_features值
        num_in_features = vgg16_bn.classifier[6].in_features  # 4096
        vgg16_bn.classifier[6] = nn.Linear(in_features=num_in_features, out_features=101)

        # 修改第一层卷积层
        first_conv = nn.Conv2d(3, 64, 5, 2, padding=2)
        vgg16_bn.features[0] = first_conv

        # 第一层卷积层、全连阶层采用xavier初始化
        def init_weights(m):
            if type(m) == nn.Linear or m == vgg16_bn.features[0]:
                torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)  预训练模型中卷积层没有，全连阶层有
        vgg16_bn.apply(init_weights)
            # init.xavier_uniform_(vgg16_bn.features[0].weight)
            # init.constant_(vgg16_bn.features[0].bias, 0.1)
        return vgg16_bn

    #resnet101
    def get_resnet101(self,model):
        # 修改平均池化层
        #model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        # 修改全连阶层
        model.fc = nn.Sequential(
            nn.Linear(2048*1*1, 1024),
            nn.Linear(1024, 101)
        )
        # 全连阶层参数赋值
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0.0)
            # 冻结未改变层的参数
            elif isinstance(m,nn.Conv2d):
               for p in m.parameters():
                   p.requires_grad=False
            elif isinstance(m,nn.BatchNorm2d):
               for p in m.parameters():
                   p.requires_grad=False
        return model

    #resnet50
    def get_resnet50(self, model):
        # 修改全连阶层
        model.fc = nn.Sequential(
            nn.Linear(2048 *1*1, 1024),
            nn.Linear(1024, 101)
        )
        # 全连阶层参数赋值
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0.0)
            # 冻结未改变层的参数
            # elif isinstance(m, nn.Conv2d):
            #     for p in m.parameters():
            #         p.requires_grad = False
            # elif isinstance(m,nn.BatchNorm2d):
            #    for p in m.parameters():
            #        p.requires_grad=False
        return model

#另一种权重初始化的方法
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

