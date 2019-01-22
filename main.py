import os
import torch
from data_loader import *
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import models
from config import opt
from torchnet import meter
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch
import torch.nn as nn

def train(**kwargs):
    opt.model = 'ResNet34'
    opt.load_latest = False
    opt.load_model_path = None
    opt._parse(kwargs)
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(128*512, 101)
    if opt.load_model_path :
        path = None
        model.load_state_dict(torch.load(path))
    model.to(opt.device)
    critertion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=opt.lr_decay, verbose=True, min_lr=5e-6, patience=5
    )
    train_data = Food(mode='train')
    val_data = Food(mode='val')
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    best_acc = 0.
    print('Epoch\tTrain loss\tTrain acc\tValid acc')
    for param in model.features.parameters():
        param.requires_grad_(False)

    for epoch in range(opt.max_epoch):
        if epoch == 20:
            for param in model.features.parameters():
                param.requires_grad_(True)
        num_total = 0
        num_correct = 0
        running_loss = 0.
        for (data, label) in tqdm(train_dataloader):
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
            num_correct += torch.sum(prediction == target.data).numpy()
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
        print('%d\t%4.3f\t\t%4.2f%%\t%4.2f%%' %
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
        num_correct += torch.sum(prediction == target.data).numpy()
    acc = 100.0*num_correct/num_total
    model.train()
    return acc


def test(**kwargs):
    pass



if __name__=='__main__':
    import fire
    fire.Fire()