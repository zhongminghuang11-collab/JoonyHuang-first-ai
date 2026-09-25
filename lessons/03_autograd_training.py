import torch

# -----------------------------
# 训练数据
# -----------------------------

x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_data = torch.tensor([2.0, 4.0, 6.0, 8.0])


# -----------------------------
# 模型参数
# -----------------------------

# 一开始模型什么都不知道
w = torch.tensor(0.0, requires_grad=True)

# 学习率
learning_rate = 0.01


# -----------------------------
# 开始训练
# -----------------------------

for step in range(1000):

    # 1. 前向计算
    prediction = w * x_data

    # 2. 计算 Loss
    loss = ((prediction - y_data) ** 2).mean()

    # 3. 自动计算梯度
    loss.backward()

    # 4. 根据梯度修改参数
    with torch.no_grad():
        w -= learning_rate * w.grad

    # 5. 清空旧梯度
    w.grad.zero_()

    # 每 100 次显示一次
    if step % 100 == 0:
        print(
            "训练次数：",
            step,
            "Loss：",
            loss.item(),
            "w：",
            w.item()
        )


# -----------------------------
# 训练结束
# -----------------------------

print()
print("训练结束")
print("模型最终学到的 w =", w.item())


# -----------------------------
# 测试没见过的数据
# -----------------------------

x_test = torch.tensor(5.0)

prediction = w * x_test

print("输入 5，模型预测 =", prediction.item())