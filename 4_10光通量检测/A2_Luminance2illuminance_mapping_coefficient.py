import cv2
import numpy as np
from t8_coordinateTransformation import T8_coordinateTransformation
from t9_luminance2illuminance import t9_luminance2illuminance  # 确保这个函数能够正确工作

def A2_Luminance2illuminance_mapping_coefficient(gray, point_illuminances, degree=2, T_inv=None):
    """
    拟合亮度值与照度值之间的映射关系。

    Args:
        gray: 灰度图像数组
        point_illuminances: [[x坐标， y坐标， 照度值], ...]
        degree: 拟合多项式的阶数
        T_inv: 逆变换矩阵（可选）

    Returns:
        coefficients: 拟合的多项式系数
    """
    luminances = []
    illuminances = []

    for point_illuminance in point_illuminances:
        x, y, illuminance = point_illuminance
        if T_inv is not None:
            img_point = T8_coordinateTransformation((x, y), T_inv)
            w, h = img_point

            if 0 <= int(h) < gray.shape[0] and 0 <= int(w) < gray.shape[1]:
                luminance = gray[int(h), int(w)]
            else:
                luminance = 0  # 或者使用其他默认值，或者记录这个错误
        else:
            luminance = 0  # 如果没有 T_inv，使用默认值

        luminances.append(luminance)
        illuminances.append(illuminance)

    # 确保输入是 NumPy 数组
    luminances = np.array(luminances)
    illuminances = np.array(illuminances)

    # 确保有有效的亮度数据
    if len(luminances) == 0 or len(illuminances) == 0:
        raise ValueError("没有有效的亮度或照度值进行拟合。")

    # 使用多项式拟合，返回多项式的系数
    coefficients = np.polyfit(luminances, illuminances, degree)

    return coefficients


if __name__ == "__main__":
    from A1_coordinates_mapping_matrix import A1_coordinates_mapping_matrix

    # 示例使用
    gray = cv2.imread('../../projectData/LED_data/task4/2.bmp', cv2.IMREAD_GRAYSCALE)
    # 示例数据，确保坐标有效
    point_illuminances = [[25, 23, 152], [33, 12, 100], [13, 25, 126], [10, 24, 223]]

    # 假设 A1_coordinates_mapping_matrix 返回一个变换矩阵和一个逆变换矩阵
    T, T_inv = A1_coordinates_mapping_matrix(gray)

    # 拟合并获取系数
    coeffs = A2_Luminance2illuminance_mapping_coefficient(gray, point_illuminances, degree=2, T_inv=T_inv)

    # 预测亮度值为 25 时的照度值
    predicted_illuminance = t9_luminance2illuminance(25, coeffs)
    print(f"Predicted illuminance for brightness 25: {predicted_illuminance}")
