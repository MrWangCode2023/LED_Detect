import cv2
import numpy as np
from common.Common import object_curve_fitting

def is_continuous_and_uniform(image):
    # 将图像转换为灰度图
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    curve_image, curve_coordinates, curve_length = object_curve_fitting(image_gray)

    if curve_image is None or curve_coordinates is None or curve_length is None:
        raise ValueError("object_curve_fitting returned None")

    # 提取非零像素坐标
    nonzero_pixels = np.column_stack(np.nonzero(curve_image))

    # 计算曲线的连续性：检查每个像素的8邻域是否有相邻的像素
    height, width = curve_image.shape
    for y, x in nonzero_pixels:
        neighborhood = curve_image[max(0, y - 1):min(y + 2, height), max(0, x - 1):min(x + 2, width)]
        if np.sum(neighborhood) < 2:  # 自己也算一个点，所以至少应该有2个点
            print(f"曲线在坐标 ({y}, {x}) 处断裂。")
            return False, False  # 曲线不连续

    # 确保图像是单通道并且类型为 uint8
    if image_gray.dtype != 'uint8':
        image_gray = image_gray.astype('uint8')

    # 计算曲线的粗细均匀性：检测骨架化前后的像素差异
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(image_gray), cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform)

    # 如果骨架的最大距离变大，说明曲线存在粗细不均匀
    if max_dist > 1.0:
        print("曲线粗细不均匀。")
        return True, False  # 曲线连续但不均匀

    return True, True  # 曲线连续且均匀

if __name__ == '__main__':
    # 读取图像并进行二值化处理
    image = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')

    if image is None:
        raise ValueError("Failed to load image")

    # 检查曲线是否连续和均匀
    continuous, uniform = is_continuous_and_uniform(image)

    if continuous:
        print("曲线是连续的。")
    else:
        print("曲线是不连续的。")

    if uniform:
        print("曲线是粗细均匀的。")
    else:
        print("曲线是粗细不均匀的。")
