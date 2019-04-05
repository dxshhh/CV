from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import sys

relu = lambda x: max(0.0, x)

ans = []
with open('tsne.in') as f:  # 需要重新打开文本进行读取
#with open('in.txt') as f:  # 需要重新打开文本进行读取
    for line2 in f:
        tmp = line2.rstrip().split()
        for i in range(len(tmp)):
            tmp[i]=relu(float(tmp[i]))
        #tmp = torch.Tensor(tmp)
        ans.append(tmp)

#ans = ans[:2000]

x = np.array(ans)
print(x.tolist())

y = TSNE(n_components=2, perplexity=10, learning_rate=50, n_iter=10000).fit_transform(x)

with open('draw_data.py', 'w') as f:
    f.write('a=')
    f.write('%s'%y.tolist())
