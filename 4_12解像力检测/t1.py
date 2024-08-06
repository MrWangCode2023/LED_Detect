import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import find_peaks

# 设置显示中文字符的字体
font_path = "C:/Windows/Fonts/simhei.ttf"  # 根据需要更改字体路径
font_prop = font_manager.FontProperties(fname=font_path)  # 创建字体属性
plt.rcParams['font.family'] = font_prop.get_name()  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示


def linear_extension_function(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊处理

    # 获取图像的高度和宽度
    height, width = gray.shape
    print(f"height: {height}, width: {width}")

    # 计算图像的中心线
    center_line = np.mean(blurred, axis=0)  # 水平平均
    max_value = np.max(center_line)
    print(f"max value: {max_value}")

    # 使用移动平均进行平滑处理
    window_size = 10  # 窗口大小
    weights = np.ones(window_size) / window_size  # 权重
    smoothed_center_line = np.convolve(center_line, weights, mode='same')  # 平滑处理

    # 查找波峰
    peaks, _ = find_peaks(smoothed_center_line)  # 查找波峰
    peak_widths = []

    # 在图像上绘制阈值点
    output_image = image.copy()

    # 检查是否找到了波峰
    if len(peaks) == 0:
        print("未检测到波峰。")
        return peak_widths  # 如果没有波峰，返回空列表

    for peak in peaks:
        # 计算当前波峰的10%高度
        height_at_peak = smoothed_center_line[peak]
        threshold = height_at_peak * 0.1  # 计算10%的高度
        print(f"波峰位置: {peak}, 10%阈值: {threshold}")

        # 计算波峰的左右边界
        left_idx = np.where(smoothed_center_line[:peak] < threshold)[0][-1] if np.any(
            smoothed_center_line[:peak] < threshold) else 0
        right_idx = peak + np.where(smoothed_center_line[peak:] < threshold)[0][0] if np.any(
            smoothed_center_line[peak:] < threshold) else width - 1

        width = right_idx - left_idx  # 计算宽度
        peak_widths.append(width)

        # 在原始图像上标记阈值点
        cv2.circle(output_image, (left_idx, height // 2), 5, (0, 255, 0), -1)  # 左侧点
        cv2.circle(output_image, (right_idx, height // 2), 5, (0, 255, 0), -1)  # 右侧点

    # 可视化结果
    plt.figure(figsize=(15, 5))  # 设置图像显示的大小

    plt.subplot(1, 3, 1)  # 创建第一个子图
    plt.title("原始图像")  # 设置子图标题
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))  # 显示原始图像并转换颜色通道
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(1, 3, 2)  # 创建第二个子图
    plt.title("平滑中心线像素值")  # 设置子图标题
    plt.plot(smoothed_center_line, color='red', label='平滑中心线')  # 绘制平滑的中心线
    plt.xlabel("像素位置")  # 设置x轴标签
    plt.ylabel("像素值")  # 设置y轴标签

    # 绘制每个波峰的阈值线
    for peak in peaks:
        plt.axhline(y=smoothed_center_line[peak] * 0.1, color='blue', linestyle='--', label='10%高度')

    plt.legend()

    plt.subplot(1, 3, 3)  # 创建第三个子图
    plt.title("波峰宽度")  # 设置子图标题
    plt.bar(range(len(peak_widths)), peak_widths, color='green')  # 绘制波峰宽度条形图
    plt.xlabel("波峰索引")  # 设置x轴标签
    plt.ylabel("宽度")  # 设置y轴标签
    plt.xticks(range(len(peak_widths)))  # 设置x轴刻度

    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示可视化结果

    return peak_widths  # 返回波峰宽度


if __name__ == "__main__":
    image_path = "../../../projectData/LED_data/4_12/1.bmp"  # 设置图像路径
    image = cv2.imread(image_path)  # 读取图像

    # 检查图像是否成功读取
    if image is None:
        print("无法读取图像，请检查路径。")  # 如果图像读取失败，输出提示信息
    else:
        # 调用函数处理图像
        widths = linear_extension_function(image)  # 调用处理函数并获取波峰宽度
        print("波峰宽度:", widths)  # 输出波峰宽度
