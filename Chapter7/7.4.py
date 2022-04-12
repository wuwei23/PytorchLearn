import sys
import numpy as np
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch

eta = 0.4 # 学习率

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)


d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()


eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()


def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    # print(v1,v2)
    return x1 - v1, x2 - v2, v1, v2


eta, gamma = 0.4, 0.5
#这里train_2d中v1 v2还是初始设成0,迭代变化
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()


eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()


'''相对于⼩批量随机梯度下降，动量法需要对每⼀个⾃变量维护⼀个同它⼀样形状的速度变量，且超参数
⾥多了动量超参数。实现中，我们将速度变量⽤更⼴义的状态变量 states 表示。'''
#使⽤⼀个来⾃NASA的测试不同⻜机机翼噪⾳的数据集来⽐较各个优化算法
def get_data_ch7(): # 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
    data = np.genfromtxt('airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
           torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

features, labels = get_data_ch7()

def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data

d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr': 0.02, 'momentum': 0.5}, features, labels)



#简洁实现
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum':0.9}, features, labels)