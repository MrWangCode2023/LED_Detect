import cv2
import numpy as np


def t1_edges(gray):
    """
    在指定的区域内进行边缘提取。

    :param gray: 输入图像（灰度图）。
    :return: 边缘提取后的图像和边缘坐标。
    """
    H, W = gray.shape  # 获取图像的高度和宽度
    box_cx, box_cy, box_H, box_w = [0.5, 0.5, 0.8, 0.5]

    # box区域
    center_x, center_y = box_cx * W, box_cy * H  # 计算中心点坐标
    h, w = int(box_H * H), int(box_w * W)  # 计算 ROI 的高度和宽度
    x, y = int(center_x - w / 2), int(center_y - h / 2)  # 将坐标转换为整数

    # 提取感兴趣区域
    ROI_Img = gray[y:y + int(h), x:x + int(w)]

    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(ROI_Img, (5, 5), 0)

    # 进行边缘提取
    edge = cv2.Canny(blurred, 25, 75)

    # 创建一个全0的图像（与输入图像同样大小），用于放置边缘
    edges_image = np.zeros_like(gray)

    # 将提取的边缘放回空白图像中
    edges_image[y:y + int(h), x:x + int(w)] = edge

    # 提取边缘坐标
    edge_coordinates1 = np.argwhere(edges_image > 0)  # 找到所有非零像素的坐标
    edge_coordinates = edge_coordinates1[:, [1, 0]]  # 转换为 (x, y)

    return edge_coordinates, edges_image