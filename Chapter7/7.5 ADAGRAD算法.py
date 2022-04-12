import sys
import numpy as np
import math
import torch
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch


def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6 # 前两项为⾃变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


eta = 0.4 # 学习率
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))


eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

#从0实现
#使⽤⼀个来⾃NASA的测试不同⻜机机翼噪⾳的数据集来⽐较各个优化算法
def get_data_ch7(): # 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
    data = np.genfromtxt('airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
           torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

features, labels = get_data_ch7()

def init_adagrad_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += (p.grad.data ** 2)
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)

d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)


#简洁实现
d2l.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
