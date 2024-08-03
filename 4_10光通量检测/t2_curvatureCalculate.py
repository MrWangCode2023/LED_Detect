import numpy as np

def t2_curvatureCalculate(points):
    """
    计算每个点的曲率，并返回最大曲率的点。
    :param points: 形状为 (N, 2) 的数组，其中 N 是点的数量，每个点的坐标为 (x, y)。
    :return: 曲率值的列表，以及最大曲率的点的坐标和曲率值。
    """
    # 创建一个空列表用于存储每个点的曲率值
    curvatures = []

    # 遍历每个点，跳过第一个和最后一个点
    for i in range(1, len(points) - 1):
        # 当前点、前一个点和后一个点
        p0 = points[i - 1]  # 前一个点
        p1 = points[i]      # 当前点
        p2 = points[i + 1]  # 后一个点

        # 计算一阶导数（切线向量）
        dx1 = p1[0] - p0[0]  # 当前点与前一个点在 x 方向的差值
        dy1 = p1[1] - p0[1]  # 当前点与前一个点在 y 方向的差值
        dx2 = p2[0] - p1[0]  # 当前点与后一个点在 x 方向的差值
        dy2 = p2[1] - p1[1]  # 当前点与后一个点在 y 方向的差值

        # 计算一阶导数的模（向量的长度）
        mag1 = np.sqrt(dx1**2 + dy1**2)  # 前一个点与当前点的切线向量的模
        mag2 = np.sqrt(dx2**2 + dy2**2)  # 当前点与后一个点的切线向量的模

        # 防止除以零
        if mag1 == 0 or mag2 == 0:
            curvatures.append(0)  # 如果切线的模为零，曲率为0
            continue

        # 计算曲率
        # 曲率公式: K = |dx1 * dy2 - dy1 * dx2| / (mag1 * mag2)
        curvature = abs(dx1 * dy2 - dy1 * dx2) / (mag1 * mag2)  # 计算当前点的曲率
        curvatures.append(curvature)  # 将计算出的曲率值添加到列表中

    # 在第一个和最后一个点处，曲率定义为0
    curvatures.insert(0, 0)  # 第一个点的曲率为0
    curvatures.append(0)      # 最后一个点的曲率为0

    # 找到前三个最大曲率及其对应的点
    top_indices = np.argsort(curvatures)[-4:]  # 获取曲率最大三个点的索引
    max_curvature_points = [points[i] for i in top_indices]  # 根据索引获取最大曲率的点
    max_curvatures = [curvatures[i] for i in top_indices]  # 根据索引获取最大曲率值
    # max_points_curvatures = zip(max_curvature_points, max_curvatures)
    # print(f"max_points_curvatures:{max_points_curvatures[0]}")

    # 找到三个最大曲率点中y坐标值最大的点(下方的拐点)
    resultpoint = max(max_curvature_points, key=lambda p: p[1])  # p[1]是y坐标

    # return curvatures, max_curvature_points, max_curvatures  # 返回曲率值列表、最大曲率的点和对应的曲率值

    return resultpoint
