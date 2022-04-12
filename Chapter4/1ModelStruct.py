import torch
from torch import nn
from collections import OrderedDict


class MLP(nn.Module):
 # 声明带有模型参数的层，这⾥声明了两个全连接层
    def __init__(self, **kwargs):
        # 调⽤MLP⽗类Block的构造函数来进⾏必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”⼀节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10) # 输出层

    # 定义模型的前向计算，即如何根据输⼊x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))
print("==============================================")


#Sequential接收⼀个⼦模块的有序字典（OrderedDict）或者⼀系列⼦模块作
#为参数来逐⼀添加 Module 的实例，⽽模型的前向计算就是将这些实例按添加的顺序逐⼀计算。
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传⼊的是⼀个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module) # add_module⽅法会将module添加进self._modules(⼀个OrderedDict)
        else: # 传⼊的是⼀些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回⼀个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            input = module(input)
        return input


net = MySequential(
    #传入model
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    )
print(net)
print(net(X))
print("==============================================")



#ModuleList 接收⼀个⼦模块的列表作为输⼊，然后也可以类似List那样进⾏append和extend操作
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1]) # 类似List的索引访问
print(net)
print("==============================================")



#ModuleDict 接收⼀个⼦模块的字典作为输⼊, 然后也可以类似字典那样进⾏添加访问操作:
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
print("==============================================")


#通过 get_constant 函数创建训练中不被迭代的参数，即常数参数
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使⽤创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复⽤全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这⾥我们需要调⽤item函数来返回标量进⾏⽐较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))
print("==============================================")



class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())
    def forward(self, x):
        return self.net(x)
net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
X = torch.rand(2, 40)
print(net)
print(net(X))