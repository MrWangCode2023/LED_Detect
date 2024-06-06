import cv2
import numpy as np


def is_continuous_and_uniform(binary_image):
    # 骨架化图像
    skeleton_image = cv2.ximgproc.thinning(binary_image)

    # 提取非零像素坐标
    nonzero_pixels = np.column_stack(np.nonzero(skeleton_image))

    # 计算曲线的连续性：检查每个像素的8邻域是否有相邻的像素
    height, width = skeleton_image.shape
    for y, x in nonzero_pixels:
        neighborhood = skeleton_image[max(0, y - 1):min(y + 2, height), max(0, x - 1):min(x + 2, width)]
        if np.sum(neighborhood) < 2:  # 自己也算一个点，所以至少应该有2个点
            print(f"曲线在坐标 ({y}, {x}) 处断裂。")
            return False, False  # 曲线不连续

    # 计算曲线的粗细均匀性：检测骨架化前后的像素差异
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(binary_image), cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform)

    # 如果骨架的最大距离变大，说明曲线存在粗细不均匀
    if max_dist > 1.0:
        print("曲线粗细不均匀。")
        return True, False  # 曲线连续但不均匀

    return True, True  # 曲线连续且均匀


if __name__ == '__main__':
    # 读取图像并进行二值化处理
    image = cv2.imread('../task1/task1_4.jpg')

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
