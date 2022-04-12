import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models  #预训练模型
import os
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''微调是迁移学习的一种常用方法，将源数据集训练的模型最后一层全连接层前面的所有层的参数复制
到目标数据集，然后将最后的全连接层的参数设置成目标数据集需要输出的类别数，并单独对最后的全连接层
做加强训练'''


#获取热狗数据集
data_dir = r'C:\Users\Wu/Datasets/HotDog'
os.listdir(os.path.join(data_dir, "hotdog")) # ['train', 'test']
#分别读取训练数据集和测试数据集中的所有图像⽂件
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
# d2l.plt.show()


#和预训练时作同样的预处理
# 指定RGB三个通道的均值和⽅差来将图像通道归⼀化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])

train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),#随即裁剪
    transforms.RandomHorizontalFlip(),#左右翻转
    transforms.ToTensor(),
    normalize
    ])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),#中心裁剪
    transforms.ToTensor(),
    normalize
    ])



#定义和初始化模型
#使用models中的resnet18模型作为预训练模型
pretrained_net = models.resnet18(pretrained=True)
print(pretrained_net)
#将预训练模型的全连接层fc更改为我们需要的类别数,更改时随机初始化
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)

#将 fc 的学习率设为已经预训练过的部分的10倍
output_params = list(map(id, pretrained_net.fc.parameters()))
#filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                      lr=lr, weight_decay=0.001)


#微调模型
def train_fine_tuning(net, optimizer, batch_size=64, num_epochs=10):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'),
                                transform=train_augs), batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'),
                                       transform=test_augs), batch_size)
    loss = torch.nn.CrossEntropyLoss()
    #释放无关内存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

train_fine_tuning(pretrained_net, optimizer)