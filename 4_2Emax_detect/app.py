import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from Img_to_luminance import img_to_luminance
from Luminance_to_illuminance import luminance_to_illuminance
from Find_max_illuminance_region import find_max_luminance_region
from Show_result_image import show_result_image

def Emax_detect(img):
    # 1. 计算亮度图像
    luminance_image = img_to_luminance(img)

    # 2. 计算照度图像
    illuminance_image = luminance_to_illuminance(luminance_image)

    # 3. 找到最大照度区域并显示结果 result = [Emax, regions_info, show_image, optimal_binary]
    Emax, regions_info, show_image, optimal_binary = find_max_luminance_region(img, luminance_image, step=1)

    show_result_image(image, show_image, luminance_image, illuminance_image)

    return show_image, luminance_image, illuminance_image


if __name__ == "__main__":
    image = cv2.imread(r"E:\workspace\Data\LED_data\task4\2.bmp")
    show_image, luminance_image, illuminance_image = Emax_detect(image)


