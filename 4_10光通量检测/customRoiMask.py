# '../../projectData/LED_data/task4/2.bmp'
import numpy as np
import cv2


import numpy as np

def CustomRoiMask(h, w, centers=None, radius=None):
    """
    创建一个圆形掩码，支持多个圆形区域。

    :param h: 掩码的高度。
    :param w: 掩码的宽度。
    :param centers: 圆心坐标列表，形如 [(x1, y1), (x2, y2), ...]。
    :param radius: 圆的半径，默认为图像宽度和高度的最小值的一半。
    :return: 圆形掩码。
    """

    # 初始化掩码
    mask = np.zeros((h, w), dtype=np.uint8)

    # 默认中心和半径
    if centers is None:
        centers = [(0.5, 0.5)]  # 默认使用图像中心
    if radius is None:
        radius = min(w, h) // 2  # 默认半径为较小值的一半

    # 遍历所有中心点，绘制圆形区域
    for center in centers:
        x, y = center[0] * w, center[1] * h
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        circular_region = dist_from_center <= radius
        mask[circular_region] = 255  # 设置掩码中的圆形区域

    # 转为三通道
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return mask



# 示例用法
if __name__ == "__main__":
    # 示例图像
    image = cv2.imread('../../projectData/LED_data/task4/2.bmp')  # 替换为你的图像路径
    h, w, _ = image.shape

    # 定义中心点和半径
    centers = [(0.5, 0.5)]  # 示例中心点
    radius = 150  # 半径

    # 创建掩码
    mask = CustomRoiMask(h, w, centers, radius)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为三通道

    interested_region = cv2.bitwise_and(image, mask)  # 按位与操作提取感兴趣区域

    # 显示结果
    cv2.imshow("Mask", mask)
    cv2.imshow("Interested Region", interested_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
