import torch
from torch import nn
from torch.nn import init



net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1)) #pytorch已进⾏默认初始化
print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

#访问模型参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))



class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        #如果⼀个 Tensor 是 Parameter ，那么它会⾃动被添加到模型的参数列表⾥
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)#不添加
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    print(name)

#data 来访问参数数值，⽤ grad 来访问参数梯度
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad) # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)


#初始化模型参数
for name, param in net.named_parameters():
    if 'weight' in name:#权重初始化
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:#偏差初始化
        init.constant_(param, val=0)
        print(name, param.data)


# ⾃定义初始化⽅法
#如 torch.nn.init.normal_ ：
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)

#令权重有⼀半概率初始化为0,有另⼀半概率初始化为[-10,-5]和[5,10]两个区间⾥均匀分布的随机数
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()
for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

#通过改变这些参数的 data 来改写模型参数值同时不会影响梯度:
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)


# 共享模型参数
#Module 类的forward 函数⾥多次调⽤同⼀个层。
# 如果我们传⼊ Sequential 的模块是同⼀个 Module 实例的话参数也是共享的
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

#在内存中，这两个线性层其实⼀个对象
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

#因为模型参数⾥包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的:
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad) # 单次梯度是3，两次所以就是6




