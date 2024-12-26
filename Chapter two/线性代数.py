import numpy as np
import torch
#降维求和

A = np.arange(20).reshape(5, 4)

print(A)
print(A.shape, A.sum())# 求所有元素的和，从而实现二维张量降维，变成一个标量

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)# 默认求和是沿着所有轴求和，也可以定义方向，如沿axis_0(沿列)求和

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)# 默认求和是沿着所有轴求和，也可以定义方向，如沿axis_1(沿行)求和

print(A.sum(axis=(0, 1))) # 结果和A.sum()相同

print(A.mean(), A.sum()/A.size)# 两种方式都可以求出张量元素均值

print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])# 同样可以求每一列均值，A.shape()=(5,4)取元组第一个数
print(A.mean(axis=1), A.sum(axis=1)/A.shape[1])#求每一行元素均值

# 求和但不降低维度

sum_A = A.sum(axis=1, keepdims=True)
print(A)
'''[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]]
 '''



print(sum_A)# 这里注意输出的是5个单元素矩阵组成一个列张量，如果没有keepdims,就会是一个向量矩阵，shape为(5,)

print(A / sum_A)# 这里前面讲过如果两个张量运算时形状不同，numpy会自动广播成形状相同的。

print(A.cumsum(axis=0))# 沿着某个轴累积求和，如此代码最后一行就是前面各行的累积求和

'''[[ 0  1  2  3]
 [ 4  6  8 10]
 [12 15 18 21]
 [24 28 32 36]
 [40 45 50 55]]
 '''



# 点积
x = torch.arange(4, dtype = torch.float32)
y = torch.ones(4, dtype = torch.float32)
print(x, y, torch.dot(x, y))

print(torch.sum(x * y))# 通过各元素相乘也可以求点积

#矩阵与向量乘积
A = torch.from_numpy(np.arange(20).reshape(5, 4)).float()# 将 NumPy 数组转换为 PyTorch 张量
print(A.shape, x.shape, torch.mv(A, x))# matrix-vector

#矩阵与矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))# matrix-multiplication

# 范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))#范数分多种，L2范数即为即为模数

print(torch.abs(u).sum())# L1范数为向量元素的绝对值之和

print(torch.norm(torch.ones((4, 9))))# Frobenius范数满足向量范数的所有性质，它就是把矩阵所有元素平方求和开根号