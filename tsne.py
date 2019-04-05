# -*- coding: utf-8 -*-
import torch
import torch.autograd
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

#echo = 12000
#freq = 100
echo = 20000
freq = 1000

#返回Tensor
def read_data():
    ans = []
    #with open('tsne.in') as f:  # 需要重新打开文本进行读取
    with open('in.txt') as f:  # 需要重新打开文本进行读取
        for line2 in f:
            tmp = line2.rstrip().split()
            for i in range(len(tmp)):
                tmp[i]=float(tmp[i])
            tmp = torch.Tensor(tmp)
            ans.append(tmp)
    #ans = torch.stack(ans)
    #ans = ans.squeeze(1)

    ans2 = []
    #with open('tsne.in') as f:  # 需要重新打开文本进行读取
    with open('in.txt') as f:  # 需要重新打开文本进行读取
        for line2 in f:
            tmp = line2.rstrip().split()
            for i in range(len(tmp)):
                tmp[i]=float(tmp[i])
            tmp = torch.Tensor(tmp)
            ans2.append(tmp)
    #ans2 = torch.stack(ans2)
    #ans2 = ans2.squeeze(1)

    ans3 = torch.zeros(len(ans),len(ans2))
    for i, x in enumerate(ans):
        for j, y in enumerate(ans2):
            ans3[i,j] = ((x-y)**2).sum()
    return ans3



def pairwise(data):
    global device
    #print("data=",data)
    N = data.size()[0]
    x_ = data**2
    x_ = x_.sum(1)
    x_ = x_.view(N, 1)
    One = torch.ones(1,N).to(device)
    tmp = x_.matmul(One)
    ans = tmp + torch.transpose(tmp,0,1)
    ans -=  2*data.matmul(torch.transpose(data,0,1))
    #print("ans=",ans)
    return ans


def init_P(data):
    x_ = pairwise(data)
    n_diagonal = x_.size()[0]
    x_ = (1 + x_).pow(-1.0)
    #print(x_)
    part = x_.sum() - n_diagonal
    ans = x_ / part
    return ans

class TSNE(nn.Module):
    def __init__(self, n_points, n_topics, P):
        self.n_points = n_points
        super(TSNE, self).__init__()
        self.logits = nn.Embedding(n_points, n_topics)
        self.P = P

    def forward(self):
        x = self.logits.weight
        x_ = pairwise(x)
        n_diagonal = x_.size()[0]
        x_ = (1 + x_).pow(-1.0)
        part = x_.sum() - n_diagonal
        x_ = x_ / part
        Loss = self.P * (torch.log(self.P) - torch.log(x_))
        #return Loss.mean()
        return Loss.sum()

    def output(self):
        filename = 'out_data.txt'
        with open(filename, 'w') as f:
            f.write('%s' % self.logits.weight)

    def draw(self):
        x_1 = []
        y_1 = []
        x_2 = []
        y_2 = []
        x_3 = []
        y_3 = []
        x = self.logits.weight
        for i in range(int(x.size()[0]/3)):
            x_1.append(float(x[i][0]))
            y_1.append(float(x[i][1]))
        for i in range(int(x.size()[0]/3),int(x.size()[0]/3)*2):
            x_2.append(float(x[i][0]))
            y_2.append(float(x[i][1]))
        for i in range(int(x.size()[0]/3)*2,x.size()[0]):
            x_3.append(float(x[i][0]))
            y_3.append(float(x[i][1]))
        plt.scatter(x_1, y_1, marker='x')
        plt.scatter(x_2, y_2, marker='o')
        plt.scatter(x_3, y_3, marker='+')
        plt.show()

    def __call__(self, *args):
        return self.forward(*args)


def train_model(optimizer,model):
    for i in range(echo):
        #print("echo=",i)
        if i%freq == 0:
            print("echo=",i)
            model.output()
            #model.draw()
        #sys.stdout.flush()
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

def main():
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    sys.stdout.flush()

    data = read_data()
    data = data.to(device)
    P = init_P(data)
    tsne = TSNE(data.size()[0],2,P)
    tsne.to(device)
    optimizer_ft = optim.SGD(tsne.parameters(), lr=0.0005, momentum=0.9)
    train_model(optimizer_ft,tsne)
    tsne.output()
    tsne.draw()

if __name__ == "__main__":
    main()
