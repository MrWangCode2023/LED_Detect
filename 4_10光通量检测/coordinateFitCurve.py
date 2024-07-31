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
    # 示例
    points = np.array([[1, 2], [2, 2.5], [3, 3.5], [4, 4], [5, 5], [6, 6.5], [7, 7]])  # 示例坐标点
    degree = 2  # 可以更改为 1、2 或更高的阶数

    coefficients = fit_polynomial(points, degree)
    print(f"拟合多项式的系数: {coefficients}")

    # 可视化拟合结果
    x_fit = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
    y_fit = np.polyval(coefficients, x_fit)  # 计算拟合多项式在 x_fit 上的值

    plt.scatter(points[:, 0], points[:, 1], color='blue', label='原始点')
    plt.plot(x_fit, y_fit, color='red', label=f'拟合多项式（阶数={degree}）')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('多项式拟合')
    plt.legend()
    plt.grid()
    plt.show()

