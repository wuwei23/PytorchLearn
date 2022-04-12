import torch

#定义模型
net = torch.nn.Linear(10, 1).cuda()
print(net)
#多GPU运算
net = torch.nn.DataParallel(net)
print(net)

#模型保存与加载
'''事实上DataParallel也是一个nn. Module，只是这个类其中有一个module就是传入的实际模型。
因此当我们调用DataParalle1后，模型结构变了(在外面加了一层而已，从8.4.1节两个输出可以对
比看出来)。所以直接加载肯定会报错的，因为模型结构对不上。
'''
#方法1
torch.save(net.module.state_dict(), "./8.4_model.pt")
new_net = torch.nn.Linear(10, 1)
new_net.load_state_dict(torch.load("./8.4_model.pt")) # 加载成功


#方法2
torch.save(net.state_dict(), "./8.4_model.pt")
new_net = torch.nn.Linear(10, 1)
new_net = torch.nn.DataParallel(new_net)
new_net.load_state_dict(torch.load("./8.4_model.pt")) # 加载成功