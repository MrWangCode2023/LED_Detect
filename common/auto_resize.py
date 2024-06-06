import cv2
import numpy as np


def auto_resize(img, new_shape=(640, 640)):
    # 当前图像的形状 [高度, 宽度]
    shape = img.shape[:2]
    # 如果新形状是整数，则将其转换为元组
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 缩放比例 (新 / 旧)
    k = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 计算调整后的大小
    new_size = int(round(shape[1] * k)), int(round(shape[0] * k))
    # 调整图像大小
    if shape[::-1] != new_size:  # 调整大小
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return img
