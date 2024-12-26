import torch 

x = torch.arange(4.0)

x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)，激活对x的追踪
x.grad  # 默认值是None，backward函数求梯度储存在这里

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
# 整个过程就是pytorch会跟踪所有关于x的操作，然后y.backward()就是计算y关于x的梯度，最后储存在x.grad里面，它本身是不储存的。

print(x.grad == 4 * x)


# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()#这里把上面x.grad == 4 * x清除掉

y = x.sum()# y == tensor(6.)
y.backward()

print(x.grad)

# 下例y是两个向量相乘，那么对于矩阵求导一般很少这么做，所以采用求和函数变为标量
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x # 输出tensor([0., 1., 4., 9.], grad_fn=<MulBackwar
y.sum().backward()# 等价于y.backward(torch.ones(len(x)))
print(x.grad)

#分离计算：如果我想把某一部分关于x的计算不被pytorch追踪
x.grad.zero_()
y = x * x
u = y.detach()# 对于Pytorch来说这是一个关于x的常数
z = u * x

z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()# 通过求和函数的方式，可以求y对x的梯度
print(x.grad == 2 * x)

#对于任意构建函数，pytorch会自动生成计算图，然后同样可以自动求导
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)# 随机生成一个标量
d = f(a)
d.backward()

print(a.grad == d/a)