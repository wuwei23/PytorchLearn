import torch
from torch import nn

#读写变量
x = torch.ones(3)
torch.save(x, 'x.pt')
x2 = torch.load('x.pt')
print(x2)

#读写tensor列表
y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)
#读写字典
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)


#读写模型
#模块模型包含在参数中 (通过model.parameters()访问)。
#state_dict 是⼀个从参数名称隐射到参数 Tesnor 的字典对象。
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())


#优化器(optim)也有⼀个state_dict，其中包含关于优化器状态以及所使⽤的超参数的信息。
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())


#保存和加载模型
#保存和加载 state_dict (推荐⽅式)
torch.save(net.state_dict(), 'state_dict.pt') # 推荐的⽂件后缀名是pt或pth
#加载:新建模型加载原定参数
net2 = MLP()
net2.load_state_dict(torch.load('state_dict.pt'))
print(net2.state_dict())















