import torch
from matplotlib import pyplot as plt
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
sys.path.append("..") # 为了导⼊上层⽬录的d2lzh_pytorch
import d2lzh_pytorch as d2l

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


#初始化模型参数
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

#参数梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# #对多维 Tensor 按维度操作
# #同⼀列（dim=0）或同⼀⾏（dim=1），在结果中保留⾏和列这两个维度（keepdim=True）
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True))
# print(X.sum(dim=1, keepdim=True))

#定义softmax函数
def softmax(X):
    X_exp = X.exp()#指数运算
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制


#定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

#gather函数使用
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
print(y_hat,y)
#将y_hat[0][0]和y_hat[1][2]作为输出，1代表输出为列，这样二维矩阵的行号自动排列，列号用y中的值。
print(y_hat.gather(1, y.view(-1, 1)))


#定义损失函数：交叉熵损失
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

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
num_epochs, lr = 5, 0.1
# 本函数已保存在d2lzh包中⽅便以后使⽤
def train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size,params=None, lr=None, optimizer=None):
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


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,batch_size, [W, b], lr)



X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels,pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])
plt.show()
