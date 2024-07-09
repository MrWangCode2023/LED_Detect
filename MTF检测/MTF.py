import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack


# 定义计算MTF的函数
def calculate_mtf(line_img):
    # 进行二维离散傅里叶变换
    f = fftpack.fft2(line_img)
    # 将频谱中心移动到图像的中心
    fshift = fftpack.fftshift(f)
    # 计算频谱的幅度谱
    magnitude_spectrum = np.abs(fshift)

    # 提取幅度谱
    magnitude_spectrum = np.log(1 + magnitude_spectrum)

    # 计算MTF（这里示例中简单取平均值作为MTF的值）
    mtf = np.mean(magnitude_spectrum)

    return mtf, magnitude_spectrum

def calculate_PSF(line_img):
    pass

def calculate_ESF(line_img):
    pass

def calculate_LSF(line_img):
    pass


def plot_mtf_curve(magnitude_spectrum, img_size):
    # 获取幅度谱的中心行
    center_row = magnitude_spectrum[magnitude_spectrum.shape[0] // 2, :]
    # 归一化处理
    mtf_curve = center_row / np.max(center_row)

    # 空间频率范围
    freq = np.fft.fftfreq(img_size)
    freq_shifted = np.fft.fftshift(freq)

    return freq_shifted, mtf_curve


def smooth_curve(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == "__main__":
    # 读取子午线像素提取的图像
    line_img = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\36.png", cv2.IMREAD_GRAYSCALE)

    # 计算MTF
    mtf_value, magnitude_spectrum = calculate_mtf(line_img)

    print("MTF Value:", mtf_value)

    # 计算MTF曲线
    img_size = line_img.shape[1]  # 图像宽度
    freq_shifted, mtf_curve = plot_mtf_curve(magnitude_spectrum, img_size)

    # 平滑MTF曲线
    mtf_curve_smooth = smooth_curve(mtf_curve, box_pts=10)  # 使用移动平均平滑曲线

    # 显示原始图像、幅度谱和MTF曲线
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(line_img, cmap='gray')
    plt.title('Line Image')

    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')

    plt.subplot(1, 3, 3)
    plt.plot(freq_shifted, mtf_curve, label='Original MTF Curve')
    plt.plot(freq_shifted, mtf_curve_smooth, label='Smoothed MTF Curve', linewidth=1)
    plt.title('MTF Curve')
    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('Contrast (MTF)')
    plt.legend()

    plt.show()
