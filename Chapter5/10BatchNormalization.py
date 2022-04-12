import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import torchvision
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#获取数据集
def get_batch_data_fashion_mnist(batch_size,resize=None):
    #获取小批量fashion_mnist数据集
    trans = []
    if resize:#扩大图像尺寸：按比例缩放
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=r'C:/Users/Wu/AppData/Roaming/mxnet/datasets/',
                        train=True, transform=transform, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=r'C:/Users/Wu/AppData/Roaming/mxnet/datasets/',
                        train=False, transform=transform, download=False)

    #读取⼩批量
    if sys.platform.startswith('win'):  # 当前系统平台为win
        num_workers = 0  # 0表示不⽤额外的进程来加速读取数据
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

'''在模型训练时，批量归⼀化利⽤⼩批量上的均值和标准差，
不断调整神经⽹络中间输出，从⽽使整个神经⽹络在各层的中间输出的数值更稳定。'''

#批量归一化层
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使⽤传⼊的移动平均所得的均值和⽅差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        #全连接层为两个元素的元组，卷积层为四个元素，包括批量、通道等
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使⽤全连接层的情况，计算特征维上的均值和⽅差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使⽤⼆维卷积层的情况，计算通道维上（axis=1）的均值和⽅差。这⾥我们需要保持
            # X的形状以便后⾯可以做⼴播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2,
                keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0,
                keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下⽤当前的均值和⽅差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和⽅差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var


'''BatchNorm 层。它保存参与求梯度和迭代的拉伸参数 gamma 和偏移参数 beta ，
同时也维护移动平均得到的均值和⽅差，以便能够在模型预测时被使⽤。 BatchNorm 实例
所需指定的 num_features 参数对于全连接层来说应为输出个数，对于卷积层来说则为输出通道数。'''
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2: #对于全连接层和卷积层来说分别为2和4
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的 traning 属性默认为true,
        # 调⽤.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, X, self.gamma,
                self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# 使⽤批量归⼀化层的LeNet
# net = nn.Sequential(
#     nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
#     BatchNorm(6, num_dims=4),
#     nn.Sigmoid(),
#     nn.MaxPool2d(2, 2), # kernel_size, stride
#     nn.Conv2d(6, 16, 5),
#     BatchNorm(16, num_dims=4),
#     nn.Sigmoid(),
#     nn.MaxPool2d(2, 2),
#     d2l.FlattenLayer(),
#
#     nn.Linear(16*4*4, 120),
#     BatchNorm(120, num_dims=2),
#     nn.Sigmoid(),
#     nn.Linear(120, 84),
#     BatchNorm(84, num_dims=2),
#     nn.Sigmoid(),
#     nn.Linear(84, 10)
#     )

#简洁实现
net = nn.Sequential(
    nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
    nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2), # kernel_size, stride
    nn.Conv2d(6, 16, 5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    d2l.FlattenLayer(),
    nn.Linear(16*4*4, 120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
    )



batch_size = 256
train_iter, test_iter = get_batch_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs = 0.001, 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


