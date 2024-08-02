import numpy as np


def T8_transform_point(point, transformation_matrix):
    # 对坐标进行Y轴进行翻转
    point1 = [point[0], -point[1]]
    point_homogeneous = np.array([*point, 1])
    transformed_point = transformation_matrix @ point_homogeneous
    # print(f"Input point: {point}, Transformed point (homogeneous): {transformed_point}")
    return transformed_point[:2]

