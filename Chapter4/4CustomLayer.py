import torch
from torch import nn



# 不含模型参数的⾃定义层
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

y = net(torch.rand(4, 8))
print(y.mean().item())


#含模型参数的⾃定义层
#⽤ ParameterList 和 ParameterDict 分别定义参数的列表和字典
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))#append\extend新增参数
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net = MyListDense()
print(net)

class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        #update() 新增参数，使⽤ keys() 返回所有键值，使⽤ items() 返回所有键值对
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增
    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

#根据传⼊的键值来进⾏不同的前向传播
x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))


net = nn.Sequential(
 MyDictDense(),
 MyListDense(),
)
print(net)
print(net(x))








