import numpy as np  # 数据处理
import pandas as pd # 数据处理
import matplotlib.pyplot as plt # 画图
import seaborn as sns # 画图
from data_creater import create_data
from Model import MLPClassifier
condition = "(x**2 + y**2) <= 1.0**2"  # 数据生成条件
create_data(n=2000,out_path='train_data.csv',condition=condition)  # 生成数据文件
create_data(n=200,out_path='val_data.csv',condition=condition)  # 生成数据文件
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
#sns.scatterplot(x="x", y="y", hue="label", data=df) #预览数据分布
#plt.show()
model = MLPClassifier(train_set=train_df, val_set=val_df,
                      Learning_rate=0.01, batch_size=20, epochs=1000)
model.fit()
prediction = model.predict()
sns.scatterplot(x=val_df['x'], y=val_df['y'], hue=prediction,data=val_df) # 预测结果可视化
plt.show()