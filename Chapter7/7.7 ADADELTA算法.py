import math
import torch
import sys
import numpy as np
sys.path.append("..")
import d2lzh_pytorch as d2l

#从0实现
#使⽤⼀个来⾃NASA的测试不同⻜机机翼噪⾳的数据集来⽐较各个优化算法
def get_data_ch7(): # 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
    data = np.genfromtxt('airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
           torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

features, labels = get_data_ch7()

#AdaDelta算法需要对每个⾃变量维护两个状态变量，即St和 delta。
def init_adadelta_states():
    s_w, s_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), \
               torch.zeros(1, dtype=torch.float32)
    delta_w, delta_b = torch.zeros((features.shape[1], 1), dtype=torch.float32), \
                       torch.zeros(1, dtype=torch.float32)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * (p.grad.data**2)
        g = p.grad.data * torch.sqrt((delta + eps) / (s + eps))
        p.data -= g
        delta[:] = rho * delta + (1 - rho) * g * g


d2l.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)



#简洁实现
d2l.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)
