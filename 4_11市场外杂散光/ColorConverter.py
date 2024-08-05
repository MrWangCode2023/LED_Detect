import numpy as np
import cv2


class ColorConverter:
    @staticmethod
    def rgb2hsv(rgb):
        """
        将RGB值转换为HSV值。

        Args:
            rgb: 一个包含R、G、B值的元组或列表，例如 (R, G, B)。

        Returns:
            hsv: 转换后的HSV值，格式为 (H, S, V)。
        """
        rgb_normalized = np.array(rgb, dtype=np.float32) / 255.0  # 归一化到(0-1)范围
        hsv = cv2.cvtColor(np.array([[rgb_normalized]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0][0]
        return tuple(hsv)

    @staticmethod
    def rgb2cie1931(rgb):
        """
        将RGB值转换为CIE 1931色彩空间。

        Args:
            rgb: 一个包含R、G、B值的元组或列表，例如 (R, G, B)。

        Returns:
            cie1931: 转换后的CIE 1931值，格式为 (X, Y, Z)。
        """
        # 使用标准的RGB到XYZ转换矩阵
        rgb_normalized = np.array(rgb, dtype=np.float32) / 255.0  # 归一化到(0-1)范围
        # 定义转换矩阵
        matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
        xyz = np.dot(matrix, rgb_normalized)
        return tuple(xyz)

    @staticmethod
    def xyz2cielab(xyz):
        """
        将RGB值转换为CIELAB值。

        Args:
            xyz: 一个包含R、G、B值的元组或列表，例如 (R, G, B)。

        Returns:
            lab: 转换后的CIELAB值，格式为 (L, a, b)。
        """

        # 将XYZ转换为CIELAB
        x, y, z = xyz
        x = x / 95.047  # D65标准
        y = y / 100.000
        z = z / 108.883

        # CIELAB转换
        def f(t):
            if t > 0.008856:
                return t ** (1 / 3)
            else:
                return (t * 7.787) + (16 / 116)

        L = max(0, (116 * f(y)) - 16)
        a = (f(x) - f(y)) * 500
        b = (f(y) - f(z)) * 200
        CIELab = (L, a, b)

        return CIELab


# 示例用法
if __name__ == "__main__":
    rgb_value = (255, 0, 0)  # 红色
    converter = ColorConverter()

    hsv_value = converter.rgb2hsv(rgb_value)
    cie1931_value = converter.rgb2cie1931(rgb_value)
    cielab_value = converter.xyz2cielab(cie1931_value)

    print(f"RGB: {rgb_value} -> HSV: {hsv_value}")
    print(f"RGB: {rgb_value} -> CIE1931: {cie1931_value}")
    print(f"XYZ: {cie1931_value} -> CIELAB: {cielab_value}")
