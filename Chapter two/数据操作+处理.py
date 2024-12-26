import os
import pandas as pd
import torch


data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only = True)) #not a number填充成剩下值的平均值


# 第一：这段是在读取列的类型，并新建一个列Alley_Pave；第二：同时如果出现NaN,会再次新建一个列Alley_NaN
#这么做的原因是如果输出的是字符串类型，机器学习无法处理，所以要转换成数字
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))

print(X, '\n', y)
#注意两个输出，通过torch使得所有数据以张量形式出现