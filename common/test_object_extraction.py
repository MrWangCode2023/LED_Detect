import cv2
import numpy as np
from collections import namedtuple
import time


############################### 目录 ####################################
"""
1 LED区域检测:                  object_extraction(image)
2. 获取目标区域像素值:            object_color_extraction(image)
3. 用窗口展示像素值的颜色:         show_LED_color(color=(0, 0, 0))
4. 目标区域曲线拟合:              object_curve_fitting(image)
5. 对图像进行自适应resize:        auto_resize(image, new_shape=(640, 640))
6. 对曲线进行等分操作:             curve_division(curve_length, curve_coordinates, num_divisions=30)
"""
############################### 1. LED区域检测 ####################################
def object_extraction(image):
    # 转换为灰度图像
    out1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用二值化
    _, binary = cv2.threshold(out1, 30, 255, cv2.THRESH_BINARY)

    # 应用高斯模糊
    out2 = cv2.GaussianBlur(binary, (5, 5), 0)

    # 找到轮廓
    contours, _ = cv2.findContours(out2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 过滤轮廓，假设我们只保留面积大于100的轮廓
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    binary = mask
    ROI_count = len(filtered_contours)

    # print("检测到的LED区域数量：", ROI_count)

    # 显示结果
    cv2.imshow("binary", binary)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered_contours, binary, ROI_count


if __name__ == "__main__":
    # 示例图像路径
    image_path = "../Data/task1/task1_12.bmp"  # 替换为实际图像路径
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or unable to read")
        exit()

    contours, binary, ROI_count = object_extraction(image)
