import numpy as np


import numpy as np

def T7_mappingMatrix(P0, P1):
    P0 = np.array(P0)
    P1 = np.array(P1)

    # 计算 P0 和 P1 之间的距离
    d = np.sqrt(np.sum((P1 - P0) ** 2))
    N0 = np.array([0, 0])  # 新坐标系中的点 N_B1
    N1 = np.array([-d, 0])  # 新坐标系中的点 N_B2，沿 x 轴负方向

    # 计算 A_vector 和 B_vector
    A_vector = P1 - P0
    B_vector = N1 - N0

    # 计算旋转角度
    theta_A = np.arctan2(A_vector[1], A_vector[0])
    theta_B = np.arctan2(B_vector[1], B_vector[0])
    theta = theta_B - theta_A

    # 创建旋转矩阵
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # 创建 Y 轴翻转矩阵
    M_flip = np.array([
        [1, 0],
        [0, -1]
    ])

    # 组合变换：先翻转 Y 轴，然后旋转
    R_combined = M_flip @ R

    # 计算平移向量
    t = N0 - R_combined @ P0  # 应用翻转和旋转后的 P0

    # 构造变换矩阵
    T = np.eye(3)
    T[:2, :2] = R_combined
    T[:2, 2] = t

    T_inv = np.linalg.inv(T)

    # 打印调试信息
    # print(f"A_vector: {A_vector}")
    # print(f"B_vector: {B_vector}")
    # print(f"Rotation angle (theta): {theta} (radians)")
    # print(f"Translation vector (t): {t}")

    return T, T_inv


if __name__ == "__main__":
    from t8_coordinateTransformation import T8_transform_point

    A1 = [258, 363]
    A2 = [261, 261]

    T, T_inv = T7_mappingMatrix(A1, A2)

    transformed_N0 = T8_transform_point(A1, T)
    print(f"P0: {A1} 被转换到新坐标系的 N0: {transformed_N0}")

    transformed_N1 = T8_transform_point(A2, T)
    print(f"P1: {A2} 被转换到新坐标系的 N1: {transformed_N1}")

    expected_N0 = np.array([0, 0])
    d = np.sqrt(np.sum((np.array(A2) - np.array(A1)) ** 2))
    expected_N1 = np.array([-d, 0])

    print(f"Expected N0: {expected_N0}, Actual N0: {transformed_N0}")
    print(f"Expected N1: {expected_N1}, Actual N1: {transformed_N1}")

