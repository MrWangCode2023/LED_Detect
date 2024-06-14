import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.Common import object_curve_fitting, object_extraction


def evaluate_and_display_curve(curve):
    if curve.curve_coordinates is not None:
        curve_image = curve.curve_image

        if curve_image is None:
            return False, False

        nonzero_pixels = np.column_stack(np.nonzero(curve_image))
        height, width = curve_image.shape
        is_continuous = True

        for y, x in nonzero_pixels:
            # 获取 (y, x) 像素的 3x3 邻域
            neighborhood = curve_image[max(0, y - 1):min(y + 2, height), max(0, x - 1):min(x + 2, width)]
            # 如果当前像素的邻域中只有它自己，说明它是孤立的点，不连续
            if np.sum(neighborhood) < 2:
                is_continuous = False
                break

        # 使用距离变换来检查线条宽度
        dist_transform = cv2.distanceTransform(cv2.bitwise_not(curve_image), cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform)
        # 如果最大距离大于 1.0，则表示曲线中存在宽度大于一个像素的部分
        is_single_pixel_width = max_dist <= 1.0

        print(f"Curve Image: {curve.curve_image is not None}")
        print(f"Curve Coordinates: {curve.curve_coordinates is not None}")
        print(f"Curve Length: {curve.curve_length}")
        print(f"Is Continuous: {is_continuous}")
        print(f"Is Single Pixel Width: {is_single_pixel_width}")

        plt.figure()
        plt.imshow(curve.curve_image, cmap='gray')
        plt.title("Fitted Curve")
        plt.show()
    else:
        print("No curve detected")

if __name__ == "__main__":
    # 读取图像
    image = cv2.imread('../../Data/LED_data/task1/task1_4.bmp')
    curve = object_curve_fitting(image)
    # 评估和显示曲线
    evaluate_and_display_curve(curve)
