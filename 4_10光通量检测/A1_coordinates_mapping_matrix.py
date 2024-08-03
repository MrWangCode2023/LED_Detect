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


def A1_coordinates_mapping_matrix(gray_image):
    # 1 边缘处理，获得边缘图像和边缘坐标
    edge_coordinates, edges_image = t1_edges(gray_image)  # 调用边缘提取函数

    # 2 计算曲率并找到最大曲率的点作为原点
    p0 = t2_curvatureCalculate(edge_coordinates)

    # 3 获得拐点左下部分边缘线的坐标
    left_edge_points = t3_getLeftEdgePoints(edge_coordinates, p0)

    # 4 计算边缘均值点p1
    p1 = t4_left_edge_mid_point(left_edge_points)

    # 7 建立两个坐标系的映射关系矩阵
    T, T_inv = T7_mappingMatrix(p0, p1)

    return T, T_inv


if __name__ == "__main__":
    # 示例使用
    gray = cv2.imread('../../projectData/LED_data/task4/2.bmp', cv2.IMREAD_GRAYSCALE)
    # gray = cv2.imread('../../projectData/LED_data/task4/6.jpg', cv2.IMREAD_GRAYSCALE)
    T, T_inv = A1_coordinates_mapping_matrix(gray)
    print(f"T:{T}, T_inv:{T_inv}")