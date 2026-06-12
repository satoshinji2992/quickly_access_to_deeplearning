import numpy as np  # 数据处理
import pandas as pd # 数据处理
import matplotlib.pyplot as plt # 画图
from pathlib import Path
from data_creater import create_data
from Model import MLPClassifier
condition = "(x**2 + y**2) <= 1.0**2"  # 数据生成条件
if not Path('train_data.csv').exists():
    create_data(n=800,out_path='train_data.csv',condition=condition,seed=42)  # 生成数据文件
if not Path('val_data.csv').exists():
    create_data(n=200,out_path='val_data.csv',condition=condition,seed=43)  # 生成数据文件
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
#sns.scatterplot(x="x", y="y", hue="label", data=df) #预览数据分布
#plt.show()
model = MLPClassifier(train_set=train_df, val_set=val_df,
                      Learning_rate=0.1, batch_size=20, epochs=1000)
model.fit()
prediction = model.predict()
plt.scatter(val_df['x'], val_df['y'], c=prediction, cmap='coolwarm', s=16) # 预测结果可视化
if plt.get_backend().lower() != 'agg':
    plt.show()
