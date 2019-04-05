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

from croploader import *


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


#_m = math.sqrt(500)


def validate(net, dataset, num):
    global device, pose_std_, pose_mean_
    net.train(False)

    sel = list(range(len(dataset)))
    random.shuffle(sel)
    sel = sel[:num]
    counter0 = 0
    counter1 = 0
    counter_pose = 0

    acc_match0 = 0
    acc_match1 = 0
    acc_pose = 0

    for idx in sel:
        data = dataset[idx]
        inputs, answers = data
        true_pose, true_match = answers
        if true_match == -1:
            continue

        inputs = inputs.to(device)

        my_match, my_pose = net(torch.stack((inputs,)))
        my_match = my_match[0]

        if true_match == 1:
            #my_pose = my_pose * pose_std_ + pose_mean_
            counter_pose += 1
            my_pose = my_pose[0]
            my_pose = my_pose * pose_std_ + pose_mean_
            my_pose = my_pose[1:]
            true_pose = true_pose[1:].to(device)
            #print(true_pose, my_pose, (my_pose - true_pose)/true_pose)
            # three cosines theorem
            cs = torch.cos((my_pose - true_pose) * math.pi / 180)
            acc_pose += float(torch.acos(cs[0] * cs[1]))

        if my_match[true_match] > my_match[1^true_match]:
            if true_match:
                acc_match1 += 1
            else:
                acc_match0 += 1
        if true_match:
            counter1 += 1
        else:
            counter0 += 1

        if counter0 + counter1 >= num:
            break

    net.train(True)

    if counter0 > 0:
        acc_match0 /= counter0
    if counter1 > 0:
        acc_match1 /= counter1
    acc_pose /= counter_pose
    acc_pose = acc_pose * 180 / math.pi
    print(acc_match0, acc_match1)
    return (acc_match0+acc_match1) / 2, acc_pose


if __name__ == "__main__":
    traindata = TrainDataset()
    n = len(traindata)
    print('n=',n)
    bs = 1000
    trainloader = DataLoader(traindata, batch_size=bs, shuffle=False, num_workers=16)
    for i, data in enumerate(trainloader):
        print('%d/%d=%.4f' % (i*bs, n, float(i*bs)/n))
        sys.stdout.flush()
