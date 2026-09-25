import torch


# -----------------------------
# 训练数据
# -----------------------------

x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])

y_data = torch.tensor([3.0, 5.0, 7.0, 9.0])


# -----------------------------
# 模型参数
# -----------------------------

# 权重
w = torch.tensor(0.0, requires_grad=True)

# 偏置
b = torch.tensor(0.0, requires_grad=True)

# 学习率
learning_rate = 0.01


# -----------------------------
# 开始训练
# -----------------------------

for step in range(1000):

    # 前向计算
    prediction = w * x_data + b

    # 计算损失
    loss = ((prediction - y_data) ** 2).mean()

    # 自动计算梯度
    loss.backward()

    # 更新参数
    with torch.no_grad():

        w -= learning_rate * w.grad

        b -= learning_rate * b.grad

    # 清空梯度
    w.grad.zero_()
    b.grad.zero_()

    # 每100次显示一次
    if step % 100 == 0:

        print(
            "训练次数：",
            step,
            "Loss：",
            loss.item(),
            "w：",
            w.item(),
            "b：",
            b.item()
        )


# -----------------------------
# 训练完成
# -----------------------------

print()

print("训练结束")

print("模型学到的 w =", w.item())

print("模型学到的 b =", b.item())


# -----------------------------
# 测试
# -----------------------------

x_test = torch.tensor(5.0)

prediction = w * x_test + b

print("输入 5，模型预测 =", prediction.item())