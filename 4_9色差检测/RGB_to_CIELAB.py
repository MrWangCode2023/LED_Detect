import numpy as np
from colormath.color_objects import LabColor

def rgb_to_xyz(rgb):
    # 转换RGB到线性RGB
    def gamma_correction(channel):
        channel = channel / 255.0
        return np.where(channel > 0.04045, ((channel + 0.055) / 1.055) ** 2.4, channel / 12.92)

    rgb = np.array(rgb)
    linear_rgb = gamma_correction(rgb)

    # 转换线性RGB到XYZ
    transformation_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    xyz = np.dot(transformation_matrix, linear_rgb)
    return xyz

def xyz_to_lab(xyz, ref_white=(0.95047, 1.00000, 1.08883)):
    # 定义参考白点
    Xn, Yn, Zn = ref_white
    x, y, z = xyz / np.array([Xn, Yn, Zn])

    # 定义辅助函数
    def f(t):
        delta = 6 / 29
        return np.where(t > delta ** 3, t ** (1/3), t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.array([L, a, b])

def rgb_to_cielab(rgb):
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    # 返回LabColor对象
    return LabColor(lab_l=lab[0], lab_a=lab[1], lab_b=lab[2])

if __name__ == "__main__":
    # 示例用法
    rgb = [255, 0, 0]  # 红色
    lab = rgb_to_cielab(rgb)
    print(f"CIELAB: {lab}")
