import numpy as np
import matplotlib.pyplot as plt
import cv2

# 生成一些模拟数据（目标区域的离散点）
np.random.seed(42)
x = np.linspace(-10, 10, 100)
y = 2 * x ** 2 + 3 * x + 5 + np.random.normal(0, 10, size=x.shape)


# 定义二次多项式的损失函数
def loss_function(params, x, y):
    a, b, c = params
    predictions = a * x ** 2 + b * x + c
    return np.mean((predictions - y) ** 2)


# 定义梯度下降法
def gradient_descent(x, y, learning_rate=0.001, epochs=1000):
    # 初始化参数
    params = np.random.randn(3)
    history = []

    for _ in range(epochs):
        a, b, c = params
        predictions = a * x ** 2 + b * x + c
        error = predictions - y

        # 计算梯度
        grad_a = np.mean(2 * error * x ** 2)
        grad_b = np.mean(2 * error * x)
        grad_c = np.mean(2 * error)

        # 更新参数
        params[0] -= learning_rate * grad_a
        params[1] -= learning_rate * grad_b
        params[2] -= learning_rate * grad_c

        # 记录损失值
        loss = loss_function(params, x, y)
        history.append(loss)

    return params, history


# 执行梯度下降法
learning_rate = 0.001
epochs = 10000
params, history = gradient_descent(x, y, learning_rate, epochs)

print(f'Optimized parameters: a={params[0]}, b={params[1]}, c={params[2]}')

# 绘制损失值变化曲线
plt.plot(history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over time')
plt.show()

# 生成单像素点的曲线图
x_fine = np.linspace(min(x), max(x), 1000)
y_fine = params[0] * x_fine ** 2 + params[1] * x_fine + params[2]

# 将坐标转换为图像坐标
img_height = 500
img_width = 500
img = np.zeros((img_height, img_width), dtype=np.uint8)

# 将曲线坐标转换为图像坐标
x_img = np.interp(x_fine, (x_fine.min(), x_fine.max()), (0, img_width - 1)).astype(int)
y_img = np.interp(y_fine, (y_fine.min(), y_fine.max()), (img_height - 1, 0)).astype(int)

# 确保每个点只占一个像素
for i in range(len(x_img)):
    img[y_img[i], x_img[i]] = 255

# 显示图像
plt.imshow(img, cmap='gray')
plt.title('Single Pixel Curve')
plt.show()
