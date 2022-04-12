import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torchvision
import torchvision.transforms as transforms

def get_batch_data_fashion_mnist(batch_size):
    #获取小批量fashion_mnist数据集
    mnist_train = torchvision.datasets.FashionMNIST(root=r'C:/Users/Wu/AppData/Roaming/mxnet/datasets/',
                        train=True, transform=transforms.ToTensor(), download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=r'C:/Users/Wu/AppData/Roaming/mxnet/datasets/',
                        train=False, transform=transforms.ToTensor(), download=False)

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

#获取小批量数据集
batch_size = 256
train_iter, test_iter = get_batch_data_fashion_mnist(batch_size)

#定义模型
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
    )
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

#损失函数
loss = torch.nn.CrossEntropyLoss()
#梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

#计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

#评价模型 net 在数据集 data_iter 上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

#训练模型
num_epochs = 5

def train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size,params=None,
              lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step() # “softmax回归的简洁实现”⼀节将⽤到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) ==y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'%
              (epoch + 1, train_l_sum / n, train_acc_sum / n,test_acc))

num_epochs = 5
#使用d2l自带的train_ch3会导致cuda出问题，因为d2l.evaluate_accuracy有判断cuda
train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size,
              None, None, optimizer)
