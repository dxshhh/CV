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

from repr2loader import *


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
            my_pose = -my_pose[1:]
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


def main():
    net = CompdNet()
    net.load_state_dict(torch.load('repr2-last.pkl'))
    print('load parameters')
    sys.stdout.flush()

    global device, pose_std_, pose_mean_
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print("device:", device)
    sys.stdout.flush()

    pose_std_ = pose_std.to(device)
    pose_mean_ = pose_mean.to(device)
    std180 = (180 - pose_mean_[1]) / pose_std_[1]
    std360 = (360 - pose_mean_[1]) / pose_std_[1]

    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)

    testdataset = TestDataset()

    #print('acc_match:%.2f, acc_pose:%.2f' % validate(net, testdataset, min(10000, len(testdataset))))
    #print('acc_pose:%.2f, acc_match:%.2f' % validate(net, testdataset, len(testdataset)))
    sys.stdout.flush()

    output_period = 100
    acc_loss_m = 0
    acc_loss_p = 0
    acc0_m = 0
    acc_ang = 0
    valid_period = 1500
    save_period = 10000
    counter = 0

    batch_size = 32

    for epoch in range(100):
        print('epoch:', epoch+1)

        traindata = TrainDataset()
        trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=100)

        for data in trainloader:
            counter += 1
            #print("train counter:", counter)

            inputs, answers = data
            true_pose, true_match = answers
            valid = true_match >= 0
            inputs = inputs[valid].to(device)
            true_match = true_match[valid].to(device)
            true_pose = true_pose[valid].to(device)

            optimizer.zero_grad()

            my_match, my_pose = net(inputs)

            valid = true_match==1
            if valid.sum().is_nonzero():
                delta_pose = my_pose[valid] - true_pose[valid]

                dpose = (delta_pose * pose_std_ + pose_mean_) * math.pi / 180
                acc_ang += torch.acos(torch.cos(dpose[:,1]) * torch.cos(dpose[:,2])).mean()

                delta_pose[delta_pose[:,1] > std180, 1] -= std360
                delta_pose[delta_pose[:,1] < -std180, 1] += std360
                pose_loss = torch.sum(delta_pose ** 2, 1)
                t = pose_loss > 1
                if t.sum() >= 1:
                    pose_loss[t] = 1 + torch.log(pose_loss[t])
                pose_loss = pose_loss.mean()
                #pose_loss = (my_pose[valid] - true_pose[valid]) * pose_std_ + pose_mean_
                #pose_loss = torch.cos(pose_loss[:,1:] * math.pi / 180)
                #pose_loss = torch.acos(pose_loss[:,0] * pose_loss[:,1])
                #pose_loss = pose_loss.mean()
            else:
                pose_loss = 0

            factor = 3
            loss = factor * F.cross_entropy(my_match, true_match)
            loss += pose_loss
            loss.backward()

            optimizer.step()
            scheduler.step()

            #acc0_m += float((torch.max(my_match,1)[1] == true_match).sum()) / batch_size
            acc_loss_p += pose_loss
            acc_loss_m += (loss - pose_loss) / factor

            if counter % output_period == 0:
                print('step:%d, acc_loss_m:%.2f, acc_loss_p:%.2f, acc_ang:%.2f' %
                        (counter, acc_loss_m / output_period, acc_loss_p / output_period, acc_ang / output_period))
                acc_loss_m = 0
                acc_loss_p = 0
                #acc0_m = 0
                acc_ang = 0
                sys.stdout.flush()

            if counter % valid_period == 0:
                acc_match, acc_pose = validate(net, testdataset, 1000)
                print('acc_match:%.2f, acc_pose:%.2f' % (acc_match, acc_pose))
                sys.stdout.flush()

            if counter % save_period == 0:
                torch.save(net.state_dict(), 'repr2-%d.pkl' % counter)
                print('saved repr2-%d.pkl' % counter)
                sys.stdout.flush()


if __name__ == "__main__":
    main()
