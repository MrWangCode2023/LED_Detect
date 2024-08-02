import cv2
import numpy as np
from t1_edges import t1_edges
from t2_curvatureCalculate import t2_curvatureCalculate
from t3_getLeftEdge import t3_getLeftEdgePoints
from t4_left_edge_mid_point import t4_left_edge_mid_point
from coordinateFitCurve import fit_polynomial
from t6_transformCoordinatesSystem import t6_transformCoordinatesSystem
from t5_drawCoordinateSystem import t5_drawCoordinateSystem
from t7_mappingMatrix import T7_mappingMatrix
from t8_coordinateTransformation import T8_transform_point
from t9_luminance2illuminance import t9_luminance2illuminance


def app(image, points):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图

    # 1 边缘处理，获得边缘图像和边缘坐标
    edges_image, edge_coordinates = t1_edges(gray_image)  # 调用边缘提取函数

    # 2 计算曲率并找到前三个最大曲率的点
    p0 = t2_curvatureCalculate(edge_coordinates)
    print(f"po:{p0}")

    # 3 获得拐点左下部分边缘线的坐标
    left_edge_points = t3_getLeftEdgePoints(edge_coordinates, p0)

    # 4 计算边缘均值点
    p1 = t4_left_edge_mid_point(left_edge_points)
    print(f"p1:{p1}")

    # 5 在原图上绘制坐标系
    t5_drawCoordinateSystem(image, p0, p1)


    # 6 建立新直角坐标系映射关系
    # transformed_points, x_axis_vector_normalized, y_axis_vector = t6_transformCoordinatesSystem(p1, p0, points=None)

    # 7 建立两个坐标系的映射关系矩阵
    T, T_inv = T7_mappingMatrix(p0, p1)

    # 8 进行坐标映射
    for point in points:
        image_coordinates = T8_transform_point(point, T_inv)
        x, y = image_coordinates
        point_brightness = gray_image[y, x]

        # 9 通过亮度值计算照度值
        illuminance = t9_luminance2illuminance(point_brightness)



    # 可视化最大曲率点
    # 创建一个空白图像
    edge_img = np.zeros_like(image)
    show_img = image.copy()

    # 绘制边缘线
    for p in edge_coordinates:
        cv2.circle(edge_img, (p[1], p[0]), 1, (255, 255, 255), -1)  # p[1]是x坐标，p[0]是y坐标
    # cv2.polylines(visualization_image, [left_edge_points], isClosed=False, color=(255, 255, 255), thickness=1)

    # 绘制y值最大的点
    cv2.circle(show_img, (p0[1], p0[0]), 1, (0, 0, 255), -1)  # 红色点
    cv2.circle(edge_img, (p0[1], p0[0]), 1, (0, 0, 255), -1)  # 红色点
    # cv2.circle(visualization_image, (p1[1], p1[0]), 1, (0, 0, 255), -1)  # 红色点

    # 显示图像
    cv2.imshow('show_img', show_img)
    cv2.imshow('edge_img', edge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return p0


if __name__ == "__main__":
    # 示例使用
    image = cv2.imread('../../projectData/LED_data/task4/2.bmp')
    # image = cv2.imread('../../projectData/LED_data/task4/6.jpg')
    app(image)

