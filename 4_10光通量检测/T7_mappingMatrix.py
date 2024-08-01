import numpy as np

import numpy as np

def mappingMatrix(P_A1, P_A2, P_B1, P_B2):
    """
    Args:
        P_A1: A坐标系中的P_A1点
        P_A2: A坐标系中的P_A2点
        P_B1: B坐标系中的P_B1点
        P_B2: B坐标系中的P_B2点
    Returns:
        T:A坐标系到B坐标系映射矩阵
        T_inv：B坐标系到A坐标系映射矩阵
    """

    # 将输入的点转换为 NumPy 数组
    P_A1 = np.array(P_A1)
    P_A2 = np.array(P_A2)
    P_B1 = np.array(P_B1)
    P_B2 = np.array(P_B2)

    # 计算坐标系 A 中的向量
    A_vector = P_A2 - P_A1
    # 计算坐标系 B 中的向量
    B_vector = P_B2 - P_B1

    # 计算旋转角度
    theta_A = np.arctan2(A_vector[1], A_vector[0])
    theta_B = np.arctan2(B_vector[1], B_vector[0])
    theta = theta_B - theta_A

    # 计算平移向量
    translation_vector = P_B1 - P_A1

    # 计算旋转矩阵
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # 构造变换矩阵
    T = np.eye(3)  # 3x3 单位矩阵
    T[:2, :2] = R  # 填入旋转矩阵
    T[:2, 2] = translation_vector  # 填入平移向量

    # 计算变换矩阵的逆
    T_inv = np.eye(3)
    T_inv[:2, :2] = R.T  # 逆的旋转矩阵为旋转矩阵的转置
    T_inv[:2, 2] = -np.dot(R.T, translation_vector)  # 逆的平移向量

    return T, T_inv

def transform_point(point, transformation_matrix):
    # 将点转换为齐次坐标
    point_homogeneous = np.array([*point, 1])
    # 使用变换矩阵进行转换
    transformed_point = transformation_matrix @ point_homogeneous
    return transformed_point[:2]  # 返回二维坐标

# 示例点
P_A1 = (1, 2)
P_A2 = (4, 6)
P_B1 = (3, 5)
P_B2 = (7, 9)

# 计算变换矩阵及其逆
transformation_matrix, inverse_matrix = mappingMatrix(P_A1, P_A2, P_B1, P_B2)

# 从坐标系 A 转换到 B
point_A = (2, 3)  # 需要转换的点
point_B = transform_point(point_A, transformation_matrix)
print(f"坐标系 A 中的点 {point_A} 转换到坐标系 B 中的点 {point_B}")

# 从坐标系 B 转换回 A
point_B2 = (5, 7)  # 需要转换的点
point_A2 = transform_point(point_B2, inverse_matrix)
print(f"坐标系 B 中的点 {point_B2} 转换回坐标系 A 中的点 {point_A2}")

