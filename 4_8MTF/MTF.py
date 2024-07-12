import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_data(data, sigma=2):
    """
    使用高斯滤波对数据进行平滑处理。

    参数：
    data: 输入数据数组。
    sigma: 高斯滤波的标准差。默认为2，决定平滑的程度。

    返回：
    平滑后的数据数组。
    """
    return gaussian_filter1d(data, sigma=sigma)


def esf(pixel_values):
    """
    计算并返回归一化的边缘扩展函数（ESF），最大值为255。

    参数：
    pixel_values: 输入像素值数组。

    返回：
    归一化并平滑后的ESF数组。
    """
    esf1 = pixel_values  # 将输入的像素值赋给 esf1 变量。
    esf2 = smooth_data(esf1)  # 对像素值进行平滑处理，得到 ESF。

    # 归一化处理，使 ESF 的最大值为 255
    esf = esf2 / np.max(esf2)

    return esf


def lsf(esf):
    """
    计算并返回线扩展函数（LSF）。

    参数：
    esf: 边缘扩展函数（ESF）数组。

    返回：
    平滑后的LSF数组。
    """
    lsf1 = np.diff(esf, axis=0)  # 对 ESF 进行差分计算，得到 LSF。
    lsf2 = smooth_data(lsf1)  # 对 LSF 进行平滑处理。
    lsf = lsf2 / np.max(lsf2)
    return lsf


def mtf(lsf):
    """
    计算并返回调制传递函数（MTF）。

    参数：
    lsf: 线扩展函数（LSF）数组。

    返回：
    平滑后的MTF数组。
    """
    mtf1 = np.abs(np.fft.fft(lsf, axis=0))  # 对 LSF 进行快速傅里叶变换，并取其绝对值，得到 MTF。
    # mtf1 = mtf / np.max(mtf)  # 将 MTF 归一化，即将其最大值归为1。
    mtf2 = smooth_data(mtf1)  # 对归一化后的 MTF 进行平滑处理。
    mtf = mtf2 / np.max(mtf2)
    return mtf


# 示例代码
if __name__ == "__main__":
    # 假设 pixel_values 是已经读取并预处理好的像素值数组
    pixel_values = [10, 20, 30, 40, 50, 40, 30, 20, 10]
    esf_result = esf(pixel_values)
    print("ESF结果：", esf_result)
