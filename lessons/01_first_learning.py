# -----------------------------
# Aurora 第一个会“学习”的程序
# -----------------------------

# 训练数据
x_data = [1, 2, 3, 4]
y_data = [2, 4, 6, 8]

# 模型一开始什么都不知道
# 所以我们把参数 w 设置成 0
w = 0.0

# 学习速度
learning_rate = 0.01


# 训练 1000 次
for step in range(1000):

    total_loss = 0.0
    gradient = 0.0

    # 查看每一组训练数据
    for x, y in zip(x_data, y_data):

        # 模型进行预测
        prediction = w * x

        # 预测与正确答案之间的差距
        error = prediction - y

        # 计算错误程度
        total_loss += error ** 2

        # 计算应该怎样修改 w
        gradient += 2 * error * x


    # 求平均
    total_loss = total_loss / len(x_data)
    gradient = gradient / len(x_data)

    # 修改模型参数
    w = w - learning_rate * gradient


    # 每训练 100 次显示一次
    if step % 100 == 0:
        print(
            "训练次数：",
            step,
            "损失：",
            total_loss,
            "参数 w：",
            w
        )


print()
print("训练结束")
print("模型学到的 w =", w)

# 测试一个模型从来没有见过的数字
x_test = 5

prediction = w * x_test

print("输入 5，模型预测：", prediction)