import torch

# 创建一个可以被训练的参数 w
w = torch.tensor(0.0, requires_grad=True)

# 一组训练数据
x = torch.tensor(4.0)
y = torch.tensor(8.0)

# -------------------------
# 1. 前向计算
# -------------------------

prediction = w * x

# 预测与正确答案的误差
error = prediction - y

# 损失
loss = error ** 2

print("参数 w =", w.item())
print("预测结果 =", prediction.item())
print("正确答案 =", y.item())
print("Loss =", loss.item())

# -------------------------
# 2. 反向传播
# -------------------------

loss.backward()

# PyTorch 自动计算出来的梯度
print("w 的梯度 =", w.grad.item())