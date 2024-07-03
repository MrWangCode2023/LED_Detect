import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def polynomial_fit(x, y, max_degree):
    """
    使用最小二乘法拟合多项式，并返回拟合函数、多项式字符串及拟合误差

    参数:
    x (ndarray): 自变量数据
    y (ndarray): 因变量数据
    max_degree (int): 最大多项式的阶数

    返回:
    best_coefficients (ndarray): 拟合的最佳多项式系数，从高阶到低阶
    best_poly_func (function): 拟合后的最佳多项式函数
    best_poly_str (str): 拟合后的最佳多项式字符串表示
    min_mse (float): 拟合的最佳均方误差
    best_degree (int): 最佳的多项式阶数
    mse_list (list): 每个阶数对应的均方误差列表
    """
    best_degree = 1
    min_mse = float('inf')
    mse_list = []
    best_coefficients = 0
    best_poly_func = None
    best_poly_str = ""

    for degree in range(1, max_degree + 1):
        # 构造范德蒙德矩阵
        A = np.vander(x, degree + 1)

        # 使用最小二乘法求解系数
        coefficients, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        # 创建多项式函数
        def poly_func(z):
            return sum(coef * z ** i for i, coef in enumerate(coefficients[::-1]))

        # 计算均方误差
        y_pred = poly_func(x)
        mse = mean_squared_error(y, y_pred)
        mse_list.append(mse)

        if mse < min_mse:
            min_mse = mse
            best_degree = degree
            best_coefficients = coefficients
            best_poly_func = poly_func
            best_poly_str = " + ".join([f"{coef:.2e}*x^{degree - i}" if degree - i > 0 else f"{coef:.2e}"
                                        for i, coef in enumerate(coefficients)])

        print(f'阶数: {degree}, 拟合的均方误差: {mse:.2f}')

    print(f'最佳的多项式阶数: {best_degree}')
    print(f'拟合的多项式函数: y = {best_poly_str}')
    print(f'拟合的均方误差: {min_mse:.2f}')

    # 生成拟合曲线数据
    x_fit = np.linspace(min(x), max(x), 10000)
    y_fit = best_poly_func(x_fit)

    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # 绘制拟合曲线和实际数据点
    ax1.scatter(x, y, color='red', label='Measured Data')
    ax1.plot(x_fit, y_fit, color='blue', label='Fitted Polynomial')
    ax1.set_xlabel('Luminance')
    ax1.set_ylabel('Illuminance')
    ax1.set_title('Polynomial Fitting')
    ax1.legend()
    ax1.grid(True)

    # 绘制阶数与均方误差的图像
    ax2.plot(np.arange(1, max_degree + 1), mse_list, marker='o', linestyle='-', color='b')
    ax2.set_xlabel('Degree of Polynomial')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('MSE LOSS')
    ax2.grid(True)

    # 调整子图之间的间距
    fig.tight_layout()

    plt.show()

    return best_coefficients, best_poly_func


if __name__ == "__main__":
    # 测试拟合函数
    measured_illuminance = np.array([100, 700, 300, 100, 200, 150, 500])
    measured_luminance = np.array([50, 100, 150, 200, 250, 300, 350])
    max_degree = 3  # 设定最大阶数

    coefficients, poly_func = polynomial_fit(measured_luminance, measured_illuminance, max_degree)