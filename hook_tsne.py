from __future__ import print_function
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import os
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import random

import warnings
warnings.filterwarnings("ignore")


class ReprNet(nn.Module):

    def __init__(self):
        super(ReprNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 7, 1)
        self.conv2 = nn.Conv2d(20, 40, 5, 1)
        self.conv3 = nn.Conv2d(40, 80, 4, 1)
        self.conv4 = nn.Conv2d(80, 160, 4, 2)
        self.fc1 = nn.Linear(160, 500)
        self.fc2 = nn.Linear(500, 500)
        self.norm1 = nn.BatchNorm2d(20)
        self.norm2 = nn.BatchNorm2d(40)
        self.norm3 = nn.BatchNorm2d(80)
        self.norm4 = nn.BatchNorm2d(160)

    def forward(self, x):
        p = 0.2
        # x.size() = [?, 3, 101, 101]
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = F.max_pool2d(x, (2, 2))
        #x = F.dropout(x, p)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = F.max_pool2d(x, (2, 2))
        #x = F.dropout(x, p)
        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv4(x))
        x = self.norm4(x)
        x = F.max_pool2d(x, (2, 2))
        # x.size() = [?, 160, 1, 1]
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.fc2(x)
        return x


class CompdNet(nn.Module):

    def __init__(self):
        super(CompdNet, self).__init__()
        self.rnet = ReprNet()
        self.fc = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 2)
        self.fc3 = nn.Linear(500, 3)

    def forward(self, x0):
        # x0.size() = [?, 6, 101, 101]
        x, y = torch.split(x0, 3, 1)
        x = self.rnet(x)
        y = self.rnet(y)
        x0 = torch.cat((x, y), 1)
        x0 = F.relu(self.fc(x0))
        return self.fc2(x0), self.fc3(x0)


imgtrans = transforms.Compose([
    transforms.Resize((101, 101)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

readimg = lambda path: imgtrans(Image.open(path).convert("RGB"))

class MyDataset(Dataset):
    def __init__(self, imglist):
        super(MyDataset, self).__init__()
        self.imglist = imglist
    def __len__(self):
        return len(self.imglist)
    def __getitem__(self, idx):
        name = self.imglist[idx]
        return (readimg(name), idx)

def read_input():
    with open('tsne.in0') as f:
        imglist = []
        n = int(f.readline())
        for i in range(n):
            x = f.readline()
            if x[-1] == '\n': x = x[:-1]
            imglist.append(x)
        return MyDataset(imglist)


def main():
    net0 = CompdNet()
    net0.load_state_dict(torch.load('repr2-last.pkl'))
    net = net0.rnet
    print('load parameters')

    global device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net.to(device)
    print("device:", device)
    sys.stdout.flush()

    imgdata = read_input()

    loader = DataLoader(imgdata, batch_size=8, num_workers=8)

    f = open('tsne.in', 'w')
    for i, data in enumerate(loader):
        inputs, idxs = data
        outs = net(inputs.to(device))
        for out in outs:
            f.write(" ".join([str(float(x)) for x in out]) + "\n")
        #errs = torch.sum((outs - qvec)**2, 1)
        #for idx, err in zip(idxs, errs):
        #    err0.append( (int(idx), float(err)) )
        print(i)
        sys.stdout.flush()
    f.close()


if __name__ == "__main__":
    main()
