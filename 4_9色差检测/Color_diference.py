from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000, _get_lab_color1_vector, _get_lab_color2_matrix
from colormath.color_conversions import convert_color
from colormath import color_diff, color_diff_matrix
import numpy as np

def delta_e_cie2000(color1, color2, Kl=1, Kc=1, Kh=1):
    """
    Calculates the Delta E (CIE2000) of two colors.
    """
    color1_vector = _get_lab_color1_vector(color1)
    color2_matrix = _get_lab_color2_matrix(color2)
    delta_e = color_diff_matrix.delta_e_cie2000(
        color1_vector, color2_matrix, Kl=Kl, Kc=Kc, Kh=Kh)[0]
    return delta_e.item()

def calculate_delta_e_lab(color1_lab, color2_lab):
    """
    计算两个LAB颜色之间的色差（CIEDE2000算法）。

    参数:
    - color1_lab: 第一个LAB颜色，LabColor对象
    - color2_lab: 第二个LAB颜色，LabColor对象

    返回:
    - delta_e: 色差值，浮点数
    """
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return delta_e

if __name__ == "__main__":
    # 示例用法
    # 假设有两个LAB颜色
    color1_lab = LabColor(lab_l=50, lab_a=0, lab_b=0)
    color2_lab = LabColor(lab_l=60, lab_a=5, lab_b=-10)

    # 计算色差
    delta_e = calculate_delta_e_lab(color1_lab, color2_lab)

    # 打印结果
    print(f"CIEDE2000色差值: {delta_e}")
