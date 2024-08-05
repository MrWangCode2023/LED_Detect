import cv2
import numpy as np


def t5_drawCoordinateSystem(image, p0=[0, 0], p1=[0, 0]):
    # po:[204 223]
    # p1:[204, 170]

    # 定义 p0 和 p1
    p0 = np.array(p0)  # 替换为实际的 p0 坐标
    p1 = np.array(p1)  # 替换为实际的 p1 坐标

    # 计算向量
    vector = p0 - p1
    length = np.linalg.norm(vector)
    unit_vector = vector / length

    # 确定坐标轴方向
    x_dir = unit_vector
    y_dir = np.array([unit_vector[1], -unit_vector[0]])  # 逆时针旋转 90 度

    # 定义坐标轴长度
    L = 200  # 可以根据需要调整长度

    # 计算坐标轴的端点
    x_axis_start = p0 - L * x_dir
    x_axis_end = p0 + L * x_dir
    y_axis_start = p0 - L * y_dir
    y_axis_end = p0 + L * y_dir

    # 在图像上绘制坐标轴
    cv2.line(image, tuple(x_axis_start.astype(int)), tuple(x_axis_end.astype(int)), (255, 0, 0), 1)  # 蓝色 x 轴
    cv2.line(image, tuple(y_axis_start.astype(int)), tuple(y_axis_end.astype(int)), (255, 0, 0), 1)  # 蓝色 y 轴

    # 在 p0 和 p1 上绘制点
    cv2.circle(image, tuple(p0), 1, (0, 0, 255), -1)  # 红色 p0


if __name__ == "__main__":
    # 读取图像
    image = cv2.imread('../../projectData/LED_data/task4/6.jpg')  # 替换为图像的路径
    t5_drawCoordinateSystem(image, point0=[204, 223], point1=[204, 170])

    # 显示结果
    cv2.imshow('Image with Coordinate System', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()