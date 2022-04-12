import torch
import torch.utils.data as Data
import torch.nn as nn
import numpy as np
from torch.nn import init
import torch.optim as optim

#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size())) #噪声


#读取数据
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取⼩批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break


#定义模型：实现一个线性回归模型
#nn 的核⼼数据结构是 Module ，它是⼀个抽象概念，既可以表示神经⽹络中的某个层（layer），
# 也可以表示⼀个包含很多层的神经⽹络。
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net.linear) # 使⽤print可以打印出⽹络的结构

# #⽤ nn.Sequential 来更加⽅便地搭建⽹络
# 写法⼀
# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
#     # 此处还可以传⼊其他层
#     )
# # 写法⼆
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module ......
# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#     ('linear', nn.Linear(num_inputs, 1))
#     # ......
#     ]))
# print(net)
# print("net[0]:  ",net[0])
#
# #查看模型所有的可学习参数
# for param in net.parameters():
#     print(param)




#初始化模型参数
# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0) # 也可以直接修改bias的data:net[0].bias.data.fill_(0)

#使用自定义模型LinearNet
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)


#损失函数
loss = nn.MSELoss()



#优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
# #为不同子网络设置不同的学习率
# optimizer =optim.SGD([
#     # 如果对某个参数不指定学习率，就使⽤最外层的默认学习率
#     {'params': net.subnet1.parameters()}, # lr=0.03
#     {'params': net.subnet2.parameters(), 'lr': 0.01}
#     ], lr=0.03)
# # 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
#


#训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        #这里涉及到多个类型转换
        output = net(X.type(torch.FloatTensor))
        l = loss(output.type(torch.DoubleTensor), y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


dense = net.linear
print(true_w, dense.weight)
print(true_b, dense.bias)