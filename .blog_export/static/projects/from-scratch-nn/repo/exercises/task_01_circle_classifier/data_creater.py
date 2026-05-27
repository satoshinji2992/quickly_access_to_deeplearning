import numpy as np
import pandas as pd

def create_data(n=1000, variance=6.0,
                 out_path="data.csv",
                 condition="((x**2 + y**2) <= 1.0**2)"):
    # 独立的两个正态分布：均值0，方差variance（协方差为0）
    samples = np.random.normal(loc=0.0, scale=np.sqrt(variance), size=(n, 2))
    x = samples[:, 0]
    y = samples[:, 1]
    labels = (eval(condition)).astype(int)
    df = pd.DataFrame({"x": x, "y": y, "label": labels})
    df.to_csv(out_path, index=False)
