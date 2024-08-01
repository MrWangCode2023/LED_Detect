import cv2
import numpy as np
import matplotlib.pyplot as plt


def fit_polynomial(points, degree=1):
    """
    根据给定的坐标点拟合一条多项式曲线，并返回拟合曲线的参数。
    :param points: 形状为 (N, 2) 的数组，其中 N 是点的数量，每个点的坐标为 (x, y)。
    :param degree: 多项式的阶数。
    :return: 拟合多项式的系数。
    """
    # 提取 x 和 y 坐标
    x = points[:, 0]
    y = points[:, 1]

    # 使用指定阶数的多项式进行拟合
    coefficients = np.polyfit(x, y, degree)

    return coefficients

if __name__ == "__main__":
    import cv2
    import numpy as np
    from t1_edges import t1_edges
    from t2_curvatureCalculate import t2_curvatureCalculate
    from t3_getLeftEdge import t3_getLeftEdgePoints
    from coordinateFitCurve import fit_polynomial
    # 示例使用
    image = cv2.imread('../../projectData/LED_data/task4/2.bmp')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图

    # 边缘处理，获得边缘图像和边缘坐标
    edges_image, edge_coordinates = t1_edges(gray_image)  # 调用边缘提取函数

    # 计算曲率并找到前三个最大曲率的点
    resultpoint = t2_curvatureCalculate(edge_coordinates)

    # 找到三个最大曲率点中y坐标值最大的点(下方的拐点)
    # resultpoint = max(max_points, key=lambda p: p[0])  # p[0]是y坐标

    # 获得拐点左下部分边缘线的坐标
    left_edge_points = t3_getLeftEdgePoints(edge_coordinates, resultpoint)

    # 拟合左边边缘线
    coefficients = fit_polynomial(left_edge_points, degree=1)
    print(f"拟合多项式的系数: {coefficients}")

    # 可视化拟合结果
    x_fit = np.linspace(min(left_edge_points[:, 0]), max(left_edge_points[:, 0]), 100)
    y_fit = np.polyval(coefficients, x_fit)  # 计算拟合多项式在 x_fit 上的值

    plt.scatter(left_edge_points[:, 0], left_edge_points[:, 1], color='blue', label='origin point')
    plt.plot(x_fit, y_fit, color='red', label=f'degree=1')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title('多项式拟合')
    plt.legend()
    plt.grid()
    plt.show()

