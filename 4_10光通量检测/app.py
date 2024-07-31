import cv2
import numpy as np
from edges import edges
from curvatureCalculate import curvatureCalculate
from getLeftEdge import getLeftEdgePoints
from coordinateFitCurve import fit_polynomial


def app(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图

    # 边缘处理，获得边缘图像和边缘坐标
    edges_image, edge_coordinates = edges(gray_image)  # 调用边缘提取函数

    # 计算曲率并找到前三个最大曲率的点
    curvatures, max_points, max_curvatures = curvatureCalculate(edge_coordinates)

    # 找到三个最大曲率点中y坐标值最大的点(下方的拐点)
    resultpoint = max(max_points, key=lambda p: p[0])  # p[0]是y坐标
    left_edge_points = getLeftEdgePoints(edge_coordinates, resultpoint)

    # 拟合左边边缘线
    coefficients = fit_polynomial(left_edge_points, degree=1)


    # 打印最大曲率点及其值
    for i in range(3):
        print(f"Max Curvature Point {i + 1}: {max_points[i]}, Curvature Value: {max_curvatures[i]:.4f}")

    print(f"Lowest Point: {resultpoint}")

    # 可视化最大曲率点
    # 创建一个空白图像
    visualization_image1 = np.zeros_like(image)
    visualization_image = image.copy()

    # 绘制边缘线
    for p in left_edge_points:
        cv2.circle(visualization_image1, (p[1], p[0]), 1, (255, 255, 255), -1)  # p[1]是x坐标，p[0]是y坐标

    # 绘制y值最大的点
    cv2.circle(visualization_image, (resultpoint[1], resultpoint[0]), 1, (0, 0, 255), -1)  # 红色点

    # 显示图像
    cv2.imshow('Lowest Curvature Point', visualization_image)
    cv2.imshow('visualization_image1', visualization_image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resultpoint


if __name__ == "__main__":
    # 示例使用
    image = cv2.imread('../../projectData/LED_data/task4/2.bmp')
    # image = cv2.imread('../../projectData/LED_data/task4/6.jpg')
    app(image)

