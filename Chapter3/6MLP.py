import torch
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


#设置参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs,num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens,num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

#激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

#定义模型
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

#损失函数
loss = torch.nn.CrossEntropyLoss()

#训练模型
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size, params, lr)


