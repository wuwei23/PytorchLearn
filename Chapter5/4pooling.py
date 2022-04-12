import torch
from torch import nn


#池化（pooling）层的提出是为了缓解卷积层对位置的过度敏感性
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))#最大池化层
print(pool2d(X, (2, 2),'avg'))#平均池化层


#填充和步幅
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))#前两个维度分别是批量和通道
print(X)
pool2d = nn.MaxPool2d(3)#不服和池化窗口形状相同，都为3
print(pool2d(X))
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))#非正方形池化窗口
print(pool2d(X))

#多通道
#池化层对每个输⼊通道分别池化，⽽不是像卷积层那样将各通道的输⼊按通道相加
X = torch.cat((X, X + 1), dim=1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
