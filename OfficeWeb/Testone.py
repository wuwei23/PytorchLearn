from __future__ import print_function
import torch
from torch.autograd import Variable

# x = torch.rand(5,3)
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
#     x = x.to(device)                       # or just use strings ``.to("cuda")``
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

x=Variable(torch.ones(2,2),requires_grad=True)
print (x)
y=x+2
print(y)

print (x.grad_fn)
print (y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

out.backward()
print(x.grad)

print("=========================")
x = torch.tensor([2., 1.], requires_grad=True).view(1, 2)
y = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

z = torch.mm(x, y) #矩阵相乘
print(f"z:{z}")
z.backward(torch.Tensor([[1., 0]]), retain_graph=True)
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")