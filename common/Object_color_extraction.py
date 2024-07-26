from collections import namedtuple

import cv2
import numpy as np

from common.Common import show_object_color, object_extraction, show_image


def object_color_extraction(image):
    ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
    filtered_contours, binary, object_count = object_extraction(image)
    object_color_image = cv2.bitwise_and(image, image, mask=binary)
    # 计算掩码图像中非零像素的数量
    nonzero_pixel_count = float(np.count_nonzero(binary))

    # 通道拆分
    blue_channel = object_color_image[:, :, 0]
    green_channel = object_color_image[:, :, 1]
    red_channel = object_color_image[:, :, 2]

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

    color_image = show_object_color(ROI_BGR_mean)
    image_dict = {
        "Image": image,
        "Object_color_image": object_color_image,
        "Color": color_image,
    }
    show_image(image_dict)

    return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, object_count)

if __name__ == '__main__':
    img = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')
    data = object_color_extraction(img)
    print(data.ROI_count)
