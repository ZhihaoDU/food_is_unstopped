import os
import torch
from hpcp_loader import *
from torch.utils.data import DataLoader
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
    pass


def test(**kwargs):
    
    pass


def vali(model, dataloader, epoch):
    pass

if __name__=='__main__':
    import fire
    fire.Fire()