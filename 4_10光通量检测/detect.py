import numpy as np
import cv2
from t8_coordinateTransformation import T8_coordinateTransformation
from t7_mappingMatrix import T7_mappingMatrix
from t9_luminance2illuminance import t9_luminance2illuminance

def detect(gray, points, T_inv, coefficients):
    """

    Args:
        gray:
        points: 检测点位
        T_inv: 坐标转换矩阵
        coefficients: 亮度-照度映射系数
    Returns:
        illuminances: 计算得到的照度值
    """
    illuminances = []

    for p in points:
        # 新坐标系坐标转映射成图像坐标系
        # T, T_inv = T7_mappingMatrix(P0, P1)
        pt = T8_coordinateTransformation(p, T_inv)
        # 计算当前点的亮度
        luminance = gray[pt[1], pt[0]]
        # 计算照度值
        illuminance = t9_luminance2illuminance(luminance, coefficients)
        illuminances.append(illuminance)

    return illuminances


if __name__ == "__main__":
    from A1_coordinates_mapping_matrix import A1_coordinates_mapping_matrix
    from A2_Luminance2illuminance_mapping_coefficient import A2_Luminance2illuminance_mapping_coefficient

    # 示例使用
    gray = cv2.imread('../../projectData/LED_data/task4/2.bmp', cv2.IMREAD_GRAYSCALE)
    # gray = cv2.imread('../../projectData/LED_data/task4/6.jpg', cv2.IMREAD_GRAYSCALE)

    T, T_inv = A1_coordinates_mapping_matrix(gray)
    coefficients = A2_Luminance2illuminance_mapping_coefficient(gray, point_illuminances, degree=2, T_inv=None)
    illuminances = detect(gray, points, T_inv, coefficients)



