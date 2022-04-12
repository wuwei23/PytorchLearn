import torch

#创建⼀个 Tensor 并设置 requires_grad=True :
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)
y = x + 2
print(y)
print(y.grad_fn)
#x这种直接创建的称为叶⼦节点，叶⼦节点对应的 grad_fn 是 None
print(x.is_leaf, y.is_leaf) # True False
print('========================================')
#通过 .requires_grad_() 来⽤in-place的⽅式改变 requires_grad 属性
a = torch.randn(2, 2) # 标准分布 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)

print('========================================')
z = y * y * 3
out = z.mean()
print(z, out)

out.backward() # 等价于 out.backward(torch.tensor(1.))
print(x.grad)

# grad在反向传播过程中是累加的(accumulated)，这意味着每⼀次运⾏反向传播，梯度都会累
# 加之前的梯度，所以⼀般在反向传播之前需把梯度清零。
# 再来反向传播⼀次，注意grad是累加的
out2 = x.sum()
out2.backward()#这里累加之前的4.5
print(x.grad)
out3 = x.sum()
x.grad.data.zero_()#归零
out3.backward()
print(x.grad)


print('========================================')
# 在 y.backward() 时，如果 y 是标量，则不需要为 backward() 传⼊任何参数
# 否则，需要传⼊⼀个与 y 同形的 Tensor
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
#y不是⼀个标量，在调⽤ backward 时需传⼊⼀个和 y 同形的权重向量进⾏加权求和得到⼀个标量
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)

print('========================================')
#终端梯度追踪
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True
#y2的梯度不会回传，不能调⽤ y2.backward()
y3.backward()
print(x.grad)


print('========================================')
#修改 tensor 的数值，但是⼜不希望被 autograd 记录
x = torch.ones(1,requires_grad=True)
print(x.data) # 还是⼀个tensor
print(x.data.requires_grad) # 但是已经是独⽴于计算图之外
y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播
y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)
