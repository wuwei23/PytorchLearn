import torch
import time
import sys
from torch import nn, optim
import torchvision
import torch.nn.functional as F
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#获取数据集
def get_batch_data_fashion_mnist(batch_size,resize=None):
    #获取小批量fashion_mnist数据集
    trans = []
    if resize:#扩大图像尺寸：按比例缩放
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=r'C:/Users/Wu/AppData/Roaming/mxnet/datasets/',
                        train=True, transform=transform, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root=r'C:/Users/Wu/AppData/Roaming/mxnet/datasets/',
                        train=False, transform=transform, download=False)

    #读取⼩批量
    if sys.platform.startswith('win'):  # 当前系统平台为win
        num_workers = 0  # 0表示不⽤额外的进程来加速读取数据
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,drop_last=True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,drop_last=True)

    return train_iter, test_iter



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#通道注意力机制
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):#缩小比率
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        # after avg_pool
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        #通道缩小再增加，此过程增强权重大的通道
        for i in range(len(gate_channels) - 2):
            # fc->bn
            self.gate_c.add_module('gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]))
            self.gate_c.add_module('gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]))
            self.gate_c.add_module('gate_c_relu_%d'%(i+1), nn.ReLU())
        # final_fc
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        # Global avg pool
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        # C∗H∗W -> C*1*1 -> C*H*W
        # print(avg_pool.size(),in_tensor.size())
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

#空间注意力机制
class SpatiaGate(nn.Module):
    # 空间注意力得到单通道的tensor
    # dilation value and reduction ratio, set d = 4 r = 16
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatiaGate, self).__init__()
        self.gate_s = nn.Sequential()
        # 1x1 + (3x3)*2 + 1x1
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                                             kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))  # 1×H×W

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)

class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatiaGate(gate_channel)

    def forward(self, in_tensor):
        att = 1 + torch.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor





#残差块实现
#ResNet的直接映射的加入，保证了 l+1 层的网络一定比 l 层包含更多的图像信息。
class Residual(nn.Module): # 本类已保存在d2lzh_pytorch包中⽅便以后使⽤
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:#是否使⽤额外的 1*1 卷积层来修改通道数
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        # 在模型训练中使用Y+X，这样，训练出来的Y就是残差
        return F.relu(Y + X)


#RESNET模型实现
#残差块
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第⼀个模块的通道数同输⼊通道数⼀致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    blk.append(BAM(out_channels))
    return nn.Sequential(*blk)


net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) #GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))


# X = torch.rand((1, 1, 224, 224))
# for name, layer in net.named_children():
#     X = layer(X)
#     print(name, ' output shape:\t', X.shape)
# print(net)
batch_size = 64
# 如出现“out of memory”的报错信息，可减⼩batch_size或resize
train_iter, test_iter = get_batch_data_fashion_mnist(batch_size, resize=96)
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

#
