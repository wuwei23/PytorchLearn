import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


d2l.set_figsize()
img = Image.open('girlfriend.jpg')
d2l.plt.imshow(img)#显示图片


#绘图函数
def show_images(imgs, num_rows, num_cols, scale=2.0):
    '''
    将多个图像表示成num_rows * num_cols个子图
    :param imgs: 图像列表
    :param scale: 规模
    :return: 图像子图
    '''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

#分成八个子图并按2*4排列
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols,scale)

#随机左右翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())
# d2l.plt.show()
#随机左右翻转
apply(img, torchvision.transforms.RandomVerticalFlip())
# d2l.plt.show()

#随机裁剪出一块面积为原面积10% ~ 100%的区域，且该区域的宽和高之比随机取自0.5~ 2,
# 然后再将该区域的宽和高分别缩放到200像素
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale= (0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
# d2l.plt.show()

#颜色变化：亮度 brightness 、对⽐度 contrast 、饱和度 saturation 和⾊调 hue
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
# d2l.plt.show()
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))
# d2l.plt.show()
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
# d2l.plt.show()

# 叠加多个图像增⼴⽅法
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                       color_aug,shape_aug])
apply(img, augs)
# d2l.plt.show()



#使⽤图像增⼴训练模型：使⽤CIFAR-10数据集

all_imges = torchvision.datasets.CIFAR10(train=True,root=r'C:\Users\Wu/Datasets/CIFAR/', download=False)
# all_imges的每⼀个元素都是(image, label)
show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8)
# d2l.plt.show()

#应用图像增广后tensor
flip_aug = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor()])
#不适用图像增广，只将数据变成tensor
no_aug = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

num_workers = 0 if sys.platform.startswith('win32') else 4
#加载数据集
def load_cifar10(is_train, augs, batch_size,root=r'C:\Users\Wu/Datasets/CIFAR/'):
    dataset = torchvision.datasets.CIFAR10(root=root,train=is_train, transform=augs, download=False)
    return DataLoader(dataset, batch_size=batch_size,shuffle=is_train, num_workers=num_workers)


#训练模型
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)

train_with_data_aug(flip_aug, no_aug)