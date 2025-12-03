import numpy as np #导入必要的运行库
import pandas as pd

#np.random.seed(42)      #设置随机数种子，保证每次运行结果一致（可选）

a  = np.random.randn(2)

df = pd.read_csv('Salary_Data.csv') #你可以通过pandas读取本文件夹内的数据，也可以自行构建数据或是使用kaggle的数据集，教程在实践引导中
learning_rate = 0.001 #学习率
epochs = 2000 #迭代次数

if_normalize=1 #如果x和y都进行标准化，则设置if_normalize=1
if if_normalize==1: #标准化，即 (x-均值)/标准差，将x，y化为0均值，1方差分布
    df_unnormalized = df.copy()
    df['YearsExperience'] = (df['YearsExperience']-df['YearsExperience'].mean())/df['YearsExperience'].std()
    df['Salary'] = (df['Salary']-df['Salary'].mean())/df['Salary'].std()

for i in range(epochs):
    loss_vector = df['Salary']-a[0]*df['YearsExperience']-a[1]
    loss = np.sum(loss_vector**2)/len(loss_vector)  #损失函数


    gradient_a0 = -2*np.mean((df['Salary']-a[0]*df['YearsExperience']-a[1])*df['YearsExperience']) #用数学方法计算梯度
    gradient_a1 = -2*np.mean(df['Salary']-a[0]*df['YearsExperience']-a[1])


    a[0] = a[0] - learning_rate*gradient_a0 #更新参数
    a[1] = a[1] - learning_rate*gradient_a1

    if i % 100 == 0: #每100个epoch打印一次损失
        print(f'Epoch {i}, Loss: {loss}')

if if_normalize==1: #如果x和y都进行标准化，则需要将参数a进行还原
    a[0] = a[0] * df_unnormalized['YearsExperience'].std() + df_unnormalized['YearsExperience'].mean()
    a[1] = a[1] * df_unnormalized['Salary'].std() + df_unnormalized['Salary'].mean()

print(a)
print(loss)