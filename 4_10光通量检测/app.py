import cv2
import numpy as np
from t1_edges import t1_edges
from t2_curvatureCalculate import t2_curvatureCalculate
from t3_getLeftEdge import t3_getLeftEdgePoints
from t4_left_edge_mid_point import t4_left_edge_mid_point
from t5_drawCoordinateSystem import t5_drawCoordinateSystem
from t7_mappingMatrix import T7_mappingMatrix
from t8_coordinateTransformation import T8_coordinateTransformation
from t9_luminance2illuminance import t9_luminance2illuminance


def app(image, points):
    luminances = []
    illuminances = []
    show_img = image.copy()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图


###################### 建立坐标系映射 ################################
    # 1 边缘处理，获得边缘图像和边缘坐标
    edge_coordinates, edges_image = t1_edges(gray_image)  # 调用边缘提取函数

    # 2 计算曲率并找到前三个最大曲率的点
    p0 = t2_curvatureCalculate(edge_coordinates)

    # 3 获得拐点左下部分边缘线的坐标
    left_edge_points = t3_getLeftEdgePoints(edge_coordinates, p0)

    # 4 计算边缘均值点p1
    p1 = t4_left_edge_mid_point(left_edge_points)

    # 7 建立两个坐标系的映射关系矩阵
    T, T_inv = T7_mappingMatrix(p0, p1)

    # 8 进行坐标映射
    for point in points:
        img_point = T8_coordinateTransformation(point, T_inv)
        x, y = img_point

############################## 获取坐标点亮度值 ################################
        luminance = gray_image[int(y), int(x)]
        luminances.append(luminance)



        # 9 通过亮度值计算照度值
        illuminance = t9_luminance2illuminance(luminance)
        illuminances.append(illuminance)


    # 可视化
    # 5 在原图上绘制坐标系
    t5_drawCoordinateSystem(show_img, p0, p1)
    # 绘制边缘线
    for p in edge_coordinates:
        cv2.circle(show_img, (p[0], p[1]), 1, (255, 255, 255), -1)  # p[1]是x坐标，p[0]是y坐标
    # 绘制y值最大的点
    cv2.circle(show_img, (p0[0], p0[1]), 1, (0, 0, 255), -1)  # 红色点



    return illuminances, show_img


if __name__ == "__main__":
    # 示例使用
    # image = cv2.imread('../../projectData/LED_data/task4/2.bmp')
    # image = cv2.imread('../../projectData/LED_data/task4/6.jpg')
    image = cv2.imread('../../../projectData/LED_data/task4/2.bmp')
    illuminances, show_img = app(image, points=[[0, 0]])

    # 打印结果
    print(f"照度值:{illuminances}")

    # 显示图像
    cv2.imshow('show_img', show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

