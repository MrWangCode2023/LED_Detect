import numpy as np


def t4_left_edge_mid_point(left_edge_points):
    # 4 计算边缘均值点
    # 提取 x 和 y 坐标
    x, y = left_edge_points[:, 0], left_edge_points[:, 1]
    # 计算 x 和 y 坐标的均值
    mean_x, mean_y = int(np.mean(x)), int(np.mean(y))
    p1 = [mean_x, mean_y]

    return p1
