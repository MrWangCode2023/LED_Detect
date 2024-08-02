import numpy as np
import matplotlib.pyplot as plt

def fit_brightness_illuminance(brightness_values, illuminance_values, degree=2):
    """
    拟合亮度值与照度值之间的映射关系。

    Args:
        brightness_values: 一维数组，表示亮度值
        illuminance_values: 一维数组，表示照度值
        degree: 拟合多项式的阶数

    Returns:
        coefficients: 拟合的多项式系数
        fit_function: 用于预测照度值的函数
    """
    # 确保输入是 NumPy 数组
    brightness_values = np.array(brightness_values)
    illuminance_values = np.array(illuminance_values)

    # 使用多项式拟合
    coefficients = np.polyfit(brightness_values, illuminance_values, degree)

    # 定义拟合函数
    def luminance2illuminance(x):
        return np.polyval(coefficients, x)



    return coefficients, luminance2illuminance


if __name__ == "__main__":
    # 示例数据
    brightness = [10, 20, 30, 40, 50]
    illuminance = [1, 4, 9, 16, 25]

    # 拟合并获取系数与拟合函数
    coeffs, luminance2illuminance = fit_brightness_illuminance(brightness, illuminance, 2)

    # 使用拟合函数进行预测
    predicted_illuminance = luminance2illuminance(25)  # 预测亮度值为 25 时的照度值
    print(f"Predicted illuminance for brightness 25: {predicted_illuminance}")
    # 可选：绘制拟合结果
    plt.scatter(brightness, illuminance, label='Data Points')
    x_fit = np.linspace(min(brightness), max(brightness), 100)
    y_fit = luminance2illuminance(x_fit)
    plt.plot(x_fit, y_fit, color='red', label=f'Polynomial Fit (degree=2)')
    plt.xlabel('Brightness Values')
    plt.ylabel('Illuminance Values')
    plt.title('Brightness vs Illuminance')
    plt.legend()
    plt.grid()
    plt.show()
