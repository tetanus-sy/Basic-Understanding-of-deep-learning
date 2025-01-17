import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from matplotlib import font_manager

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 30

# 加载支持中文的字体
font_path = 'C:/Windows/Fonts/SIMKAI.TTF'  # 替换为您的字体文件路径
font_prop = font_manager.FontProperties(fname=font_path, size=18)

plt.rc('font', family=font_prop.get_name())  # 使用正确的字体名称

animator = d2l.Animator(xlabel='周期', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['训练损失', '训练准确率', '测试准确率'])

for epoch in range(num_epochs):
    train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, trainer)
    test_acc = d2l.evaluate_accuracy(net, test_iter)
    animator.add(epoch + 1, train_metrics + (test_acc,))
    plt.pause(0.1)  # 添加暂停以展示动态图效果

def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
plt.show()