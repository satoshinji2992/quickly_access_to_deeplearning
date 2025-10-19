import numpy as np

# 1. 准备数据
# 根据教程的上下文，我们使用这些点进行训练
x_train = np.array([-2, -1, 0, 1, 2, 3, 4])
y_train = np.array([-2, -1, 0, 1, 2, 3, 4])

# 2. 初始化参数
# 随机初始化参数 a 和 b
np.random.seed(42) # 设置随机种子以保证结果可复现
a = np.random.randn()
b = np.random.randn()

# 设置超参数
learning_rate = 0.01
epochs = 1000

print(f"初始参数: a = {a:.4f}, b = {b:.4f}")

# 3. 训练模型
n = len(x_train)

for epoch in range(epochs):
    # 3.1 前向传播: 计算预测值
    y_pred = a * x_train + b

    # 3.2 计算损失 (MSE)
    loss = np.mean((y_train - y_pred)**2)

    # 3.3 计算梯度
    # loss = (1/n) * Σ(y_train - (a*x_train + b))^2
    # ∂loss/∂a = (1/n) * Σ(2 * (y_train - y_pred) * (-x_train)) = (-2/n) * Σ(x_train * (y_train - y_pred))
    # ∂loss/∂b = (1/n) * Σ(2 * (y_train - y_pred) * (-1)) = (-2/n) * Σ(y_train - y_pred)
    grad_a = (-2/n) * np.sum(x_train * (y_train - y_pred))
    grad_b = (-2/n) * np.sum(y_train - y_pred)

    # 3.4 更新参数 (反向传播)
    a = a - learning_rate * grad_a
    b = b - learning_rate * grad_b

    # 打印训练过程
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, a: {a:.4f}, b: {b:.4f}")

print("\n训练完成!")
print(f"最终学得的参数: a = {a:.4f}, b = {b:.4f}")

# 4. 预测
x_to_predict = 5
y_predicted = a * x_to_predict + b

print(f"\n当 x = {x_to_predict} 时, 预测的 y 值为: {y_predicted:.4f}")
