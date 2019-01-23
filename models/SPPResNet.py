import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math


def SPP(x, pool_size):
    N, C, H, W = x.size()
    for i in range(len(pool_size)):
        maxpool = nn.AdaptiveMaxPool2d((pool_size[i], pool_size[i]))
        if i==0: spp = maxpool(x).view(N, -1)
        else: spp = torch.cat((spp, maxpool(x).view(N, -1)),1)
    return spp


class SPPResNet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        resnet_model = torchvision.models.resnet34(pretrained=True)

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.fc = torch.nn.Linear(512*5, 101)

    def forward(self, X):

        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        x = self.conv1(X)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        X = self.layer4(x)
        #assert X.size() == (N, 512, 7, 7)
        X = SPP(X, [2,1])
        X = self.fc(X)
        return X