import numpy as np

def t3_getLeftEdgePoints(edge_points, p):
    """
    获取指定点 p 左边部分的边缘坐标。
    :param edge_points: 形状为 (N, 2) 的数组，其中 N 是边缘点的数量，每个点的坐标为 (x, y)。
    :param p: 分界点的坐标 (x, y)。
    :return: 左边部分的边缘坐标数组。
    """
    left_points = []

    # 遍历边缘点，找到所有左侧的点
    for point in edge_points:
        if point[0] < p[0]:  # 判断 x 坐标是否小于分界点的 x 坐标
            left_points.append(point)

    return np.array(left_points)


if __name__ == "__main__":
    # 示例
    edge_points = np.array([[1, 2], [3, 4], [2, 3], [5, 6], [0, 1]])  # 边缘坐标数组
    p = np.array([3, 4])  # 分界点

    left_edge_points = t3_getLeftEdgePoints(edge_points, p)
    print(left_edge_points)  # 输出左边部分的边缘坐标
