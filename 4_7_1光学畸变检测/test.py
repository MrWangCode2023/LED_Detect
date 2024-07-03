import cv2
import numpy as np

def calculate_min_bounding_rect(midpoints):
    # 将中点转换为 numpy 数组
    midpoints = np.array(midpoints, dtype=np.float32)

    # 使用 OpenCV 的 minAreaRect 计算最小外接矩形
    rect = cv2.minAreaRect(midpoints)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    return box

def draw_min_bounding_rect(image, box):
    # 绘制最小外接矩形
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)  # 红色

    return image

if __name__ == "__main__":
    # 给定的四个中点
    midpoints = [(125, 231), (317, 456), (589, 338), (394, 120)]

    # 计算最小外接矩形
    box = calculate_min_bounding_rect(midpoints)

    # 创建一个空白图像
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    # 绘制最小外接矩形
    result_image = draw_min_bounding_rect(image.copy(), box)

    # 显示结果
    cv2.imshow("Min Bounding Rectangle", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
