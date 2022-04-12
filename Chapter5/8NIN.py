import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
import torchvision
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#串联多个由卷积层和“全连接”层构成的⼩⽹络来构建⼀个深层⽹络。
'''NIN块由⼀个卷积层加两个充当全连接层的1*1卷积层串联⽽成。其中第⼀个卷
积层的超参数可以⾃⾏设置，⽽第⼆和第三个卷积层的超参数⼀般是固定的。

NiN去掉了AlexNet最后的3个全连接层，取⽽代之地，NiN使⽤了输出通道数等于标签类别数的NiN块，
然后使⽤全局平均池化层对每个通道中所有元素求平均并直接⽤于分类。
这⾥的全局平均池化层即窗⼝形状等于输⼊空间维形状的平均池化层。
NiN的这个设计的好处是可以显著减⼩模型参数尺⼨，从⽽缓解过拟合。然⽽，该设计有时会造成获得
有效模型的训练时间的增加。'''

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU())
    return blk


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗⼝形状设置成输⼊的⾼和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        #x.size()[2:]去掉通道数和批量数
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


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
                                shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    #输出为使个通道，使用全局平均池化将每个通道中所有元素求平均用于分类。
    GlobalAvgPool2d(),
    # 将四维的输出转成⼆维的输出，其形状为(批量⼤⼩, 10)
    d2l.FlattenLayer())

print(net)
# X = torch.rand(1, 1, 224, 224)
# for name, blk in net.named_children():
#     X = blk(X)
#     print(name, 'output shape: ', X.shape)



batch_size = 128
# 如出现“out of memory”的报错信息，可减⼩batch_size或resize
train_iter, test_iter = get_batch_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.001, 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)