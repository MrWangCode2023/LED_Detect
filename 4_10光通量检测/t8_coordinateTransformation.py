import numpy as np


def T8_coordinateTransformation(point, transformation_matrix):
    # 验证 transformation_matrix 是否为 3x3 矩阵
    if transformation_matrix.shape != (3, 3):
        raise ValueError("Transformation matrix must be a 3x3 matrix")

        # 对坐标进行Y轴翻转（如果需要的话）
    point1 = [point[0], -point[1]]
    point_homogeneous = np.array(point1 + [1])  # 使用列表扩展来简化代码

    # 进行矩阵乘法
    transformed_point = transformation_matrix @ point_homogeneous

    # 只返回前两个坐标
    return transformed_point[:2]


# 示例用法
if __name__ == "__main__":
    # 假设的变换矩阵（例如，一个2D旋转矩阵）
    theta = np.radians(45)  # 45度旋转
    c, s = np.cos(theta), np.sin(theta)
    transformation_matrix = np.array([[c, -s, 0],
                                      [s, c, 0],
                                      [0, 0, 1]])

    # 测试点
    point = [1, 1]
    transformed_point = T8_coordinateTransformation(point, transformation_matrix)
    print(f"Transformed point: {transformed_point}")