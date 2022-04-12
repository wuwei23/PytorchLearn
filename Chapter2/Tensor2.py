import torch

#内存开销
#索引、 view 是不会开辟新内存的，⽽像 y = x + y 这样的运算是会新开内存的，然后将 y 指向新内存。
#以使⽤Python⾃带的 id 函数看地址是否相同

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False
print('========================================')
#如果想指定结果到原来的 y 的内存，我们可以使⽤前⾯介绍的索引来进⾏替换操作。
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True
print('========================================')
#我们还可以使⽤运算符全名函数中的 out 参数或者⾃加运算符 += (也即 add_() )达到上述效果
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before) # True

print('========================================')
#TENSOR 和NUMPY相互转换
#所有在CPU上的 Tensor （除了 CharTensor ）都⽀持与NumPy数组相互转换。
#Tensor 转 NumPy
a = torch.ones(5)
b = a.numpy()
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)
#NumPy数组转 Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)
#直接⽤ torch.tensor() 将NumPy数组转换成 Tensor,会进行数据拷贝
c = torch.tensor(a)
a += 1
print(a, c)

print('========================================')
#⽤⽅法 to() 可以将 Tensor 在CPU和GPU（需要硬件⽀持）之间相互移动。
# 以下代码只有在PyTorch GPU版本上才会执⾏
if torch.cuda.is_available():
    device = torch.device("cuda") # GPU
    y = torch.ones_like(x, device=device) # 直接创建⼀个在GPU上的Tensor
    x = x.to(device) # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) # to()还可以同时更改数据类型