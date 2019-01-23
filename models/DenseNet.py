import torch, os, torchvision
from torch import nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
class BResNet(torch.nn.Module):

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

        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, 101)

        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
    def forward(self, X):

        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        x = self.conv1(X)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        X = self.layer4(x)
        assert X.size() == (N, 512, 14, 14)
        X = X.view(N, 512, 14**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 101)
        return X
    def freeze_layers(self, grad=False):
        # Freeze all previous layers.
        for param in self.conv1.parameters():
            param.requires_grad_(grad)
        for param in self.bn1.parameters():
            param.requires_grad_(grad)
        for param in self.layer1.parameters():
            param.requires_grad_(grad)
        for param in self.layer2.parameters():
            param.requires_grad_(grad)
        for param in self.layer3.parameters():
            param.requires_grad_(grad)
        for param in self.layer4.parameters():
            param.requires_grad_(grad)




class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None :
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn



class ATTDenseNet(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)
        densenet_model = torchvision.models.densenet121(pretrained=True)

        self.features = densenet_model.features
        self.multi_att = MultiHeadAttention(n_head=8, d_model=1024, d_k=64, d_v=64, dropout=0.1)

        self.fc = torch.nn.Linear(512*2, 101)

        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)


    def forward(self, X):

        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        x1 = self.features(X) # N, 1024, 7, 7
        x1 = F.relu(x1, inplace=True)
        #print(x1.shape)
        X = x1.view(N, -1, 1024)
        X, att = self.multi_att(X,X,X)
        X = X.view(N,1024, 7, 7)
        X = F.avg_pool2d(X, kernel_size=7, stride=1).view(N, -1)
        X = self.fc(X)
        assert X.size() == (N, 101)
        return X
    def freeze_layers(self, grad=False):
        # Freeze all previous layers.
        for param in self.features.parameters():
            param.requires_grad_(grad)


class DenseNet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        densenet_model = torchvision.models.densenet121(pretrained=True)
        self.features = densenet_model.features
        self.fc = torch.nn.Linear(512*2, 101)

        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X) # N, 1024, 14, 14
        X = F.relu(X, inplace=True)
        X = F.avg_pool2d(X, kernel_size=7, stride=1).view(N, -1)
        X = self.fc(X)
        assert X.size() == (N, 101)
        return X
    def freeze_layers(self, grad=False):
        # Freeze all previous layers.
        for param in self.features.parameters():
            param.requires_grad_(grad)


class BDenseNet(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)
        densenet_model = torchvision.models.densenet121(pretrained=True)
        self.features = densenet_model.features
        self.conv = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, 101)

        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
    def forward(self, X):

        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        X = self.conv(X)
        X = F.relu(X, inplace=True)
        assert X.size() == (N, 512, 7, 7)
        X = X.view(N, 512, 7**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (7**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 101)
        return X
    def freeze_layers(self, grad=False):
        # Freeze all previous layers.
        for param in self.features.parameters():
            param.requires_grad_(grad)



class SliceDenseNet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        densenet_model = torchvision.models.densenet121(pretrained=True)
        self.features = densenet_model.features
        self.fc = torch.nn.Linear(512*2, 101)
        self.conv = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.slice_conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=3, out_channels= 512,kernel_size=(5,224),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,1))),
        ]))

        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        slice = self.slice_conv(X) # N, 512, 1, 1
        X = self.features(X) # N, 1024, 14, 14
        X = self.conv(X)
        X = F.relu(X, inplace=True)
        X = F.avg_pool2d(X, kernel_size=7, stride=1)
        X = torch.cat([X, slice], 1).view(N, -1)
        X = self.fc(X)
        assert X.size() == (N, 101)
        return X
    def freeze_layers(self, grad=False):
        # Freeze all previous layers.
        for param in self.features.parameters():
            param.requires_grad_(grad)

class SSliceDenseNet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        densenet_model = torchvision.models.densenet121(pretrained=True)
        self.features = densenet_model.features
        self.fc = torch.nn.Linear(512*2, 101)
        self.conv = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.slice_conv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=3, out_channels= 256,kernel_size=(5,224),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,1))),
        ]))
        self.slice_conv1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=3, out_channels= 256,kernel_size=(224,5),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,1))),
        ]))
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        slice0 = self.slice_conv0(X) # N, 512, 1, 1
        slice1 = self.slice_conv1(X) # N, 512, 1, 1
        X = self.features(X) # N, 1024, 14, 14
        X = self.conv(X)
        X = F.relu(X, inplace=True)
        X = F.avg_pool2d(X, kernel_size=7, stride=1)
        X = torch.cat([X, slice0, slice1], 1).view(N, -1)
        X = self.fc(X)
        assert X.size() == (N, 101)
        return X
    def freeze_layers(self, grad=False):
        # Freeze all previous layers.
        for param in self.features.parameters():
            param.requires_grad_(grad)


