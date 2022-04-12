import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)
#long型全0的 Tensor
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
#数据创建
x = torch.tensor([5.5, 3])
print(x)
#依据现有tensor创建
#torch.float64对应torch.DoubleTensor
#torch.float32对应torch.FloatTensor
x = x.new_ones(5, 3, dtype=torch.float64) # 返回的tensor默认具有相同的torch.dtype和torch.device
print(x)
x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
print(x)
#看尺寸
print(x.size())
print(x.shape)

print('========================================')
print("操作形式")
#加法形式
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
#指定输出
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# adds x to y
y.add_(x)
print(y)

print('========================================')
#索引：索引出来的结果与原数据共享内存，也即修改⼀个，另⼀个会跟着修改。
y = x[0, :]
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了

print('========================================')
#⽤ view() 来改变 Tensor 的形状：
#注意 view() 返回的新tensor与源tensor共享内存（其实是同⼀个tensor），也即更改其中的⼀个，另
#外⼀个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察⻆度)
y = x.view(15)
z = x.view(-1, 5) # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
x += 1
print(x)
print(y) # 也加了1
#返回一个真正的拷贝，先⽤ clone 创造⼀个副本然后再使⽤ view
#使⽤ clone 还有⼀个好处是会被记录在计算图中，即梯度回传到副本时也会传到源 Tensor
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

print('========================================')
#item() , 它可以将⼀个标量 Tensor 转换成⼀个Python number
x = torch.randn(1)
print(x)
print(x.item())

print('========================================')
# ⼴播机制先适当复制元素使这两个 Tensor 形状相同后再按元素运算。
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)