import numpy as np
import matplotlib.pyplot as plt
from common.Get_cie1931_data import get_cie1931_data

def plot_cie_chromaticity_diagram(xy_coords, labels=None):
    """
    绘制CIE 1931色度图并在图上标注给定的xy坐标点。

    Args:
        xy_coords: 待绘制的xy坐标列表
        labels: 每个坐标点的标签列表（可选）

    Returns:
        fig, ax: matplotlib的figure和axes对象
    """
    # 获取CIE 1931数据
    cie_data = get_cie1931_data()

    # 分别提取波长、x、y、z坐标
    wavelengths = cie_data[:, 0]
    x_coords = cie_data[:, 1]
    y_coords = cie_data[:, 2]
    z_coords = cie_data[:, 3]

    # 计算x和y色度坐标，避免除以零错误
    sum_coords = x_coords + y_coords + z_coords
    valid_indices = sum_coords > 0  # 生成布尔索引数组，用于筛选有效数据

    # 初始化色度坐标数组，确保与输入数据形状相同
    x_chromaticity = np.zeros_like(x_coords)
    y_chromaticity = np.zeros_like(y_coords)

    # 仅在有效索引处计算色度坐标，避免无效值的计算
    x_chromaticity[valid_indices] = x_coords[valid_indices] / sum_coords[valid_indices]
    y_chromaticity[valid_indices] = y_coords[valid_indices] / sum_coords[valid_indices]

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制色度图曲线
    ax.plot(x_chromaticity, y_chromaticity, '-', color='black')
    ax.fill(x_chromaticity, y_chromaticity, 'gray', alpha=0.1)

    # 每隔20个数据点标注一次波长
    for i in range(0, len(wavelengths), 20):
        if valid_indices[i]:  # 确保仅标注有效数据点
            ax.text(x_chromaticity[i], y_chromaticity[i], f'{int(wavelengths[i])} nm')

    # 设置坐标轴范围和标签
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('CIE 1931 Chromaticity Diagram')

    # 在色度图上绘制给定的xy坐标点
    for i, (x, y) in enumerate(xy_coords):
        ax.plot(x, y, 'o', color='red')
        if labels:  # 如果提供了标签，则在相应位置标注
            ax.text(x, y, labels[i], fontsize=12, color='red')

    # 显示图表
    plt.show()

    return fig, ax

if __name__ == "__main__":
    # 定义待绘制的CIE xy坐标
    cie_coords = [(0.3, 0.3), (0.4, 0.4), (0.49, 0.5), (0.1, 0.7)]
    labels = ['Point 1', 'Point 2', 'Point 3', 'Point 4']  # 可选标签

    # 绘制并显示CIE坐标，返回figure和axes对象
    fig, ax = plot_cie_chromaticity_diagram(cie_coords, labels)
    # 显示图表
    plt.show()

