import os
import sys
import main
import numpy as np
import torchvision.transforms as transforms
import torch
import models

sys.path.append('../../')
from config import opt



opt.use_gpu = True
#opt._parse()
opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
opt.model, model = 'SliceDenseNet', models.SliceDenseNet()
path = '../../models/'+opt.model+'/SliceDenseNet_best.pth'
if opt.use_gpu:
    model.load_state_dict(torch.load(path))
else :
    model.load_state_dict(torch.load(path,map_location='cpu'))
model.to(opt.device)



def food_classify(img_path):
    label, score = main.test(img_path, model, opt)
    ret = []
    for li, si in zip(label, score):
        if si < 1e-4 :
            si = si + 1e-4
        ret.append({"label": int(li), "score": "%.2f" % (float(si)*100)})
    return ret


if __name__=='__main__':
    id2name=[]
    with open('../../../food-101/meta/classes.txt') as fp:
        for ii, line in enumerate(fp):
            id2name.append(line.rstrip())
    l = food_classify('../uploaded_images/cheesecake2.jpg')
    print(id2name[l[0]['label']],l[0]['score'])
