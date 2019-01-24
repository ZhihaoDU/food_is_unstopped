import os
import torch
from data_loader import *
from torch.utils.data import DataLoader
import torchvision
import models
from config import opt
#from torchnet import meter
#from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


def test(img_path, model, test_opt):
    print('img_path is '+ str(img_path))
    transform = transforms.Compose([
        transforms.Resize(size=224),  # Let smaller edge match
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    data = Image.open(img_path)
    data = data.convert('RGB')
    data = transform(data)
    data = data.unsqueeze(0)
    model.eval()
    input = data.to(test_opt.device)
    score = model(input)
    p, l = torch.sort(-score[0])
    p = torch.nn.functional.softmax(-p, dim=0)
    p = p.cpu().tolist()
    l = l.cpu().tolist()
    #print(l)
    #print(p)
    return l, p


def train(**kwargs):

    opt.load_latest = True
    #opt.load_model_path = None
    opt.lr = 1e-5
    opt.batch_size = 32
    #opt.model, model = 'ATTDenseNet', models.ATTDenseNet()
    #opt.model, model = 'BResNet', models.BResNet()SliceDenseNet
    #opt.model, model = 'DenseNet90', models.DenseNet()
    opt.model, model = 'SliceDenseNet', models.SliceDenseNet()
    #opt.model, model = 'SelfBNDenseNet', models.SelfBNDenseNet()
    #opt.model, model = 'FatSliceDenseNet', models.FatSliceDenseNet()
    opt._parse(kwargs)
    if opt.load_latest :
        path = 'models/'+opt.model+'/latest.pth'
        #model, optimizer = load_check_points(model, optimizer, path)
        model.load_state_dict(torch.load(path))
    model.to(opt.device)
    critertion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer=optimizer, mode='min', factor=opt.lr_decay, verbose=True, min_lr=5e-6, patience=3
    #)
    train_data = Food(mode='train')
    val_data = Food(mode='val')
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    best_acc = 0.
    print('Epoch\tTrain loss\tTrain acc\tValid acc')
    #model.freeze_layers(grad=False)
    for epoch in range(opt.max_epoch):
        #if epoch == 5:
        #    model.freeze_layers(grad=True)
        num_total = 0
        num_correct = 0
        running_loss = 0.
        for (data, label) in (train_dataloader):
            #train model
            data.requires_grad_()
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = critertion(score, target)
            loss.backward()
            optimizer.step()
            _, prediction = torch.max(score.data, 1)
            num_total += data.shape[0]
            num_correct += torch.sum(prediction == target.data).cpu().numpy()
            running_loss += loss.item()
        running_loss/=num_total
        acc_train = 100.0*num_correct/num_total
        #print(running_loss, acc_train)
        model_prefix = 'models/'+opt.model+'/'
        if not os.path.isdir(model_prefix):
            os.mkdir(model_prefix)
        #check_points(model, optimizer, time.strftime(model_prefix + '%m%d_%H:%M:%S.pth'))
        torch.save(model.state_dict(), time.strftime(model_prefix + '%m%d_%H:%M:%S.pth'))
        #check_points(model, optimizer, model_prefix+'latest.pth')
        torch.save(model.state_dict(), model_prefix+'latest.pth')
        scheduler.step()

        val_acc = val(model, val_dataloader, epoch)
        print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t' %
              (epoch+1, running_loss, acc_train, val_acc))
        print(time.strftime('%H:%M:%S'))

        if best_acc<val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_prefix+'best.pth')
            print('*****************BEST*****************')
    print('Best at epoch %d, test accuaray %f' % (epoch, best_acc))
'''
def check_points(model, optimizer, path):
    info = {"model": model.state_dict(),
            "optimizer": optimizer.state_dict()}
    torch.save(info, path)

def load_check_points(model, optimizer, path):
    dic = torch.load(path)
    model.load_state_dict(dic['model'])
    optimizer.load_state_dict(dic['optimizer'])
    return model, optimizer
'''
def spp_train(**kwargs):

    #opt.load_latest = True
    #opt.load_model_path = None
    opt.lr = 1e-3
    opt.batch_size=64
    #opt.model, model = 'ATTDenseNet', models.ATTDenseNet()
    #opt.model, model = 'BResNet', models.BResNet()
    opt.model, model = 'spp_ResNet', models.SPPResNet()
    opt._parse(kwargs)
    if opt.load_latest :
        path = 'models/'+opt.model+'/best.pth'
        model.load_state_dict(torch.load(path))
    model.to(opt.device)
    critertion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=opt.lr_decay, verbose=True, min_lr=5e-6, patience=5
    )
    train_data = Food(mode='train')
    train_data2 = Food(mode='train', transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(90),
                                        transforms.Resize(size=336),  # Let smaller edge match
                                        transforms.CenterCrop(size=336),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))]))
    val_data = Food(mode='val', transform=transforms.Compose([
                                            transforms.Resize(size=336),  # Let smaller edge match
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))]))
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=opt.num_workers)
    best_acc = 0.
    print('Epoch\tTrain loss\tTrain acc\tValid acc')
    model.freeze_layers(grad=False)
    for epoch in range(opt.max_epoch):
        if epoch == 5:
            model.freeze_layers(grad=True)
        num_total = 0
        num_correct = 0
        running_loss = 0.
        for (data0, label0),(data1, label1) in zip(train_dataloader,train_dataloader2):
            for flag in range(2):
                if flag is 0:
                    data, label = data0, label0
                else : data, label = data1, label1
            #train model
                data.requires_grad_()
                input = data.to(opt.device)
                target = label.to(opt.device)

                optimizer.zero_grad()
                score = model(input)
                loss = critertion(score, target)
                loss.backward()
                optimizer.step()
                _, prediction = torch.max(score.data, 1)
                num_total += data.shape[0]
                num_correct += torch.sum(prediction == target.data).cpu().numpy()
                running_loss += loss.item()
        running_loss/=num_total
        acc_train = 100.0*num_correct/num_total
        #print(running_loss, acc_train)
        model_prefix = 'models/'+opt.model+'/'
        if not os.path.isdir(model_prefix):
            os.mkdir(model_prefix)
        torch.save(model.state_dict(), model_prefix+'latest.pth')
        scheduler.step(running_loss)

        val_acc = val(model, val_dataloader, epoch)
        print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
              (epoch+1, running_loss, acc_train, val_acc))

        if best_acc<val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_prefix+'best.pth')
            print('*****************BEST*****************')
    print('Best at epoch %d, test accuaray %f' % (epoch, best_acc))

def val(model, dataloader, epoch):
    model.eval()
    num_total, num_correct = 0, 0
    for (data, label) in (dataloader):
        input = data.to(opt.device)
        target = label.to(opt.device)
        score = model(input)
        _, prediction = torch.max(score.data, 1)
        num_total += data.shape[0]
        num_correct += torch.sum(prediction == target.data).cpu().numpy()
    acc = 100.0*num_correct/num_total
    model.train()
    return acc







if __name__=='__main__':
    #import fire
    #fire.Fire()
    train()
