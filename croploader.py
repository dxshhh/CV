import os
import shutil
import math
import sys
import requests
from contextlib import closing
import torch
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image


def download_url(url, file_name):
    print('downloading %s to %s ...' % (url, file_name))
    sys.stdout.flush()
    os.system('wget -q -O %s %s' % (file_name, url))
    print('download %s finish' % url)
    sys.stdout.flush()
#    with closing(requests.get(url, stream=True)) as response:
#        chunk_size = 4096
#        content_size = int(response.headers['content-length'])
#        have_size = 0
#        with open(file_name, "wb") as f:
#            for data in response.iter_content(chunk_size=chunk_size):
#                f.write(data)
#                have_size += len(data)
#                print('download progress %.1f%%' % (float(have_size) / content_size), end='\r')
#                sys.stdout.flush()
#    print('\ndownload %s finish')


def extract_tar(fname, dname):
    print('extracting %s to %s ...' % (fname, dname))
    sys.stdout.flush()
    os.system('mkdir -p %s' % dname)
    os.system('tar xzf %s -C %s' % (fname, dname))
    print('extract %s finish' % fname)
    sys.stdout.flush()


imgtrans_withcrop = transforms.Compose([
    transforms.CenterCrop(192),
    transforms.Resize(101),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

imgtrans = transforms.Compose([
    transforms.Resize(101),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

pose_mean = torch.Tensor([0, -0.7835600899294222, -0.493369725809238])
pose_std = torch.Tensor([1, 21.81454787834064, 6.18895859105927])

stat_n = 0
stat_sum = torch.Tensor([0, 0, 0])
stat_sum2 = torch.Tensor([0, 0, 0])

class SceneDataset(Dataset):

    def __init__(self, datasetID, keepTar=True):
        super(SceneDataset, self).__init__()
        self.datasetID = datasetID
        self.path = "../data/disk/%04d.tar" % datasetID
        self.dname = os.path.dirname(os.path.abspath(self.path))
        dname2 = os.path.join(self.dname, '%04d' % datasetID)

        if not os.path.isdir(dname2):
            if not os.access(self.path, os.R_OK):
                download_url('https://storage.googleapis.com/streetview_image_pose_3d/dataset_aligned/%04d.tar' % datasetID, self.path)
            extract_tar(self.path, self.dname)

        self.dname = dname2

        self.names = []
        for i in os.listdir(self.dname):
            bname, ext = os.path.splitext(i)
            if ext == ".jpg":
                self.names.append(i)

        print('load dataset %04d finish' % datasetID)
        sys.stdout.flush()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = os.path.join(self.dname, self.names[idx])
        try:
            img = Image.open(name)
            patch_center = (img.width // 2, img.height // 2)
            patch_size_ = 192 // 2
            img = img.crop((patch_center[0]-patch_size_, patch_center[1]-patch_size_, patch_center[0]+patch_size_, patch_center[1]+patch_size_))
            img = img.resize((101, 101))
            img.save(name)
        except BaseException as e:
            print('invalid data:', e)
            sys.stdout.flush()
        return 1


class TrainDataset(Dataset):

    def __init__(self):
        super(TrainDataset, self).__init__()
        f = open('../data/list.txt', 'r')
        self.ids = [int(i) for i in f.readlines()]
        f.close()
        self.dataset = [SceneDataset(i, keepTar=True) for i in self.ids]
        self.lens = [len(d) for d in self.dataset]
        self.sumlen = sum(self.lens)

    def __len__(self):
        return self.sumlen

    def __getitem__(self, idx):
        i = 0
        while idx >= self.lens[i]:
            idx -= self.lens[i]
            i += 1
        return self.dataset[i][idx]
