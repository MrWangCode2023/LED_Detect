import numpy as np
import matplotlib.pyplot as plt

def luminance_to_illuminance(luminance_image):
    """
    将亮度图像转换为照度图像。

    参数:
    luminance_image (ndarray): 输入的亮度图像，单位为cd/m²。

    返回:
    illuminance_image (ndarray): 转换后的照度图像，单位为lux。
    """
    # 假设这里的转换系数为 0.1
    k = 0.1
    illuminance_image = luminance_image * k
    return illuminance_image

if __name__ == "__main__":
    # 示例：生成模拟的亮度图像
    luminance_image = np.random.randint(0, 255, size=(512, 512))

    # 计算照度图像
    illuminance_image = luminance_to_illuminance(luminance_image)

    # 绘制亮度图像和照度图像
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(luminance_image, cmap='gray')
    plt.title('Luminance Image')
    plt.colorbar(label='Luminance')

    plt.subplot(1, 2, 2)
    plt.imshow(illuminance_image, cmap='jet')
    plt.title('Illuminance Image')
    plt.colorbar(label='Illuminance (lux)')

    plt.tight_layout()
    plt.show()
