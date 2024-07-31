import numpy as np


def curvatureCalculate(points):
    # 创建一个列表存储每个点的曲率值
    curvatures = []
    # 存储point-curvature
    p_vs = []

    # 遍历每个点，跳过第一个和最后一个点
    for i in range(1, len(points) - 1):
        # 前一个点
        p0 = points[i - 1]
        # 当前点
        p1 = points[i]
        # 后一个点
        p2 = points[i + 1]

        # 计算当前点的一阶导数（切线向量）
        ## 当前点与前一个点在x方向的差值
        dx1 = p1[0] - p0[0]
        ## 当前点与前一个点在y方向的差值
        dy1 = p1[1] - p0[1]
        ## 当前点与后一个点在x方向的差值
        dx2 = p2[0] - p1[0]
        ## 当前点与后一个点在y方向的差值
        dy2 = p2[1] - p1[1]

        # 计算一阶导数的模（向量的长度）
        ## 前一个点与当前点的切线向量的模
        mag1 = np.sqrt(dx1**2 + dy1**2)
        ## 当前点与后一个点的切线向量的模
        mag2 = np.sqrt(dx2**2 + dy2**2)

        # 防止除零
        if mag1 == 0 or mag2 == 0:
            ## 如果切线的模为0，曲率为0
            curvatures.append(0)
            continue

        # 计算曲率[曲率公式： K = |dx1 * dy2 - dy1 * dx2| / （mag1 * mag2）]
        ## 计算当前点的曲率
        curvature = abs(dx1 * dy2 - dy1 * dx2) / (mag1 * mag2)
        p_v = (p1, curvature)
        curvatures.append(curvature)
        p_vs.append(p_v)

    # 在第一个和最后一个point处，曲率定义为0
    ## 第一个点
    curvatures.insert(0, 0)
    p_vs.insert(0, (points[0], 0))
    ## 最后一个点
    curvatures.append(0)
    p_vs.append((points[-1], 0))

    # 获取曲率最大的前四个点
    top_indices = np.argsort(curvatures)[-4]  # 返回曲率最大的四个点的索引数组
    max_curvatures_points = [points[i] for i in top_indices]  # 根据索引获取最大曲率的点坐标
    max_curvatures = [curvatures[i] for i in top_indices]  # 根据索引获取最大曲率的点的曲率值

    max_points_curvatures = zip(max_curvatures_points, max_curvatures)

    return curvatures, max_curvatures_points, max_curvatures


