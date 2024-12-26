import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

#代替原有生成数据集部分
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

# nn是神经网络的缩写neural network
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))# 表示容器nn.Sequential中存入了一个接收2 个输入特征，并输出 1 个特征的线性层

'''这就是初始化w和b'''
net[0].weight.data.normal_(0, 0.01)# 线性层中第一部分的权重w数据值替换为随机正态
net[0].bias.data.fill_(0)#  偏置初始值改为b=0

#计算损失函数
loss = nn.MSELoss()

#定义优化算法，这里SGD和MSR都是直接嵌入的
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)