import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line

# 生成一个简单的斜条测试图像
def generate_slanted_edge(width, height, angle_deg, line_thickness=2):
    # 创建空白图像
    image = np.zeros((height, width), dtype=np.uint8)

    # 计算斜条的起始和结束点
    x1, y1 = 50, 50  # 起始点
    x2, y2 = width - 50, height - 50  # 结束点

    # 绘制斜条
    rr, cc = line(y1, x1, y2, x2)
    image[rr, cc] = 255  # 设置斜条像素值为白色

    return image

# 生成斜条测试图像
test_image = generate_slanted_edge(512, 512, 45)
cv2.imwrite("E:\\workspace\\Data\\LED_data\\task4\\36.png", test_image)

# 显示测试图像
plt.figure(figsize=(6, 6))
plt.imshow(test_image, cmap='gray')
plt.title('Slanted-edge Test Chart')
plt.axis('off')
plt.show()

# 接下来可以使用生成的测试图像进行MTF计算
