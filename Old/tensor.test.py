import numpy as np
import torch

# 创建张量
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1, 2], [3, 4]])
c = torch.rand(2, 3)

print('a:', a)
print('b:', b)
print('c:', c)

print('a.shape:', a.shape)
print('b.shape:', b.shape)
print('c.shape:', c.shape)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print('x+y:', x + y)
print('x-y:', x - y)
print('x*y:', x * y)
print('x/y:', x / y)

m1 = torch.rand(2, 3)
m2 = torch.rand(3, 2)

print('m1@m2:', m1 @ m2)

x1 = torch.randn(2, 3)
print('x1=', x1)

x2 = torch.zeros(2, 3)
print('x2=', x2)

x3 = torch.ones(2, 3)
print('x3=', x3)

x4 = torch.empty(2, 3)
print('x4=', x4)

x5 = torch.arange(0, 10, 2)
print('x5=', x5)

x6 = torch.linspace(0, 1, 5)
print('x6=', x6)

x7 = torch.eye(3)
print('x7=', x7)

x8 = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
print('x8=', x8)

tensor_2d = torch.rand(2, 3)
tensor_3d = torch.stack([tensor_2d, tensor_2d + 10, tensor_2d - 5])
print("tensor_2d:", tensor_2d)
print("tensor_3d:", tensor_3d)
