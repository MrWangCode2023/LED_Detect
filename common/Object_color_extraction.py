import cv2
import numpy as np
from collections import namedtuple
from common.Object_extraction import object_extraction

def show_LED_color(color=(0, 0, 0)):
    # 定义图像的宽度和高度
    width = 640
    height = 640
    # 创建一个纯色图像，大小为 width x height，数据类型为 uint8
    color_image = np.full((height, width, 3), color, dtype=np.uint8)
    return color_image

def Object_color_extraction(image):
    ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
    filtered_contours, binary, ROI_count = object_extraction(image)
    roi_color_image = cv2.bitwise_and(image, image, mask=binary)
    # 计算掩码图像中非零像素的数量
    nonzero_pixel_count = float(np.count_nonzero(binary))

    # 通道拆分
    blue_channel = roi_color_image[:, :, 0]
    green_channel = roi_color_image[:, :, 1]
    red_channel = roi_color_image[:, :, 2]

    blue_sum = np.sum(blue_channel)
    green_sum = np.sum(green_channel)
    red_sum = np.sum(red_channel)

    # 三个通道区域的像素分别求均值
    ROI_BGR_mean = ()  # 空元组
    ROI_BGR_mean += (blue_sum / nonzero_pixel_count,)
    ROI_BGR_mean += (green_sum / nonzero_pixel_count,)
    ROI_BGR_mean += (red_sum / nonzero_pixel_count,)

    # BGR均值转换为HSV格式
    bgr_mean = np.array([[ROI_BGR_mean]], dtype=np.uint8)
    ROI_HSV_mean = cv2.cvtColor(bgr_mean, cv2.COLOR_BGR2HSV)[0][0]
    brightness = ROI_HSV_mean[2]

    return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count)

"""对于 HSV 均值颜色 [ 28 244 230]：
色调（Hue），表示颜色在颜色轮上的位置，取值范围是0到179，对应于0到360度的色相。
饱和度（Saturation），表示颜色的纯度或鲜艳度，取值范围是0到255。
亮度（Value），表示颜色的亮度，取值范围是0到255。
"""

if __name__ == '__main__':
    img = cv2.imread('../Data/task1/task1_12.bmp')
    data = Object_color_extraction(img)
    color_image = show_LED_color(data.ROI_BGR_mean)
    cv2.imshow("color_image", color_image)
    print(data.ROI_count)
    cv2.waitKey(0)
    cv2.destroyAllWindows()