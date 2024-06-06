import cv2
import numpy as np

def show_LED_color(color=(0, 0, 0)):
    # 定义图像的宽度和高度
    width = 640
    height = 640
    # 创建一个纯色图像，大小为 width x height，数据类型为 uint8
    color_image = np.full((height, width, 3), color, dtype=np.uint8)
    return color_image