import numpy as np


def t9_luminance2illuminance(luminance, coefficients=None):
    """
    将光度转换为照度。

    参数:
    luminance (float or np.ndarray): 光度值。
    coefficients (list or np.ndarray, optional): 多项式系数，按降幂排列。如果未提供，则使用简化的转换方法。

    返回:
    float or np.ndarray: 转换后的照度值。
    """
    if coefficients is not None:
        illuminance = np.polyval(coefficients, luminance)
    else:
        coefficients = [0.1, 0]
        # illuminance1 = luminance * 0.1
        illuminance = np.polyval(coefficients, luminance)
        # print(f"1:{illuminance1}  2:{illuminance}")

    # 打印函数
    degree = len(coefficients) - 1  # 计算多项式的阶数
    function_str = "f(x) = " + " + ".join(f"{coef:.2f}x^{degree - i}" for i, coef in enumerate(coefficients))
    print(function_str)
    return illuminance