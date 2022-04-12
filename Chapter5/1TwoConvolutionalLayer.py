import torch
from torch import nn


#二维卷积的互相关运算：卷积核和输入矩阵的运算
def corr2d(x, k):
    h, w = k.shape #卷积核的尺寸
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))#输出尺寸初始化
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i: i+ h, j: j + w] * k).sum()#矩阵相乘求和
    return y

#验证
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))

#自定义二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


#卷积检测图像物体边缘
#定义图像
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)
#构造卷积核
k = torch.tensor([[1,-1]])
print(k)
#横向相邻元素相同，输出为0；否则输出为⾮0。可以看到结果，横向边缘被标记
Y = corr2d(X,k)
print(Y)


# 通过数据学习核数组
# 构造⼀个核数组形状是(1, 2)的⼆维卷积层
conv2d = Conv2D(kernel_size=(1, 2))

step = 20
lr = 0.01
#核数组可以通过递归下降学习
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))



