import torch
from torch import nn


# 定义⼀个函数来计算卷积层。它对输⼊和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量⼤⼩和通道数均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:]) # 排除不关⼼的前两维：批量和通道
# 注意这⾥是两侧分别填充1⾏或列，所以在两侧⼀共填充2⾏或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)

# 使⽤⾼为5、宽为3的卷积核。在⾼和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)


conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 5))
print(comp_conv2d(conv2d, X).shape)

