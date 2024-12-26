import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from matplotlib import font_manager

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]# 此神经网络只有三层

# 加载支持中文的字体
font_path = 'C:/Windows/Fonts/SIMKAI.TTF'  # 替换为您的字体文件路径
font_prop = font_manager.FontProperties(fname=font_path, size=18)

plt.rc('font', family=font_prop.get_name())  # 使用正确的字体名称


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

animator = d2l.Animator(xlabel='周期', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['训练损失', '训练准确率', '测试准确率'])
for epoch in range(num_epochs):
    train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
    test_acc = d2l.evaluate_accuracy(net, test_iter)
    animator.add(epoch + 1, train_metrics + (test_acc,))
    plt.pause(0.1)  # 添加暂停以展示动态图效果

d2l.predict_ch3(net, test_iter)
plt.show()
