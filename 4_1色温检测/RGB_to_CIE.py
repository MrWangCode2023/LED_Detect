import numpy as np

def rgb_to_cie_xy(rgb):
    # sRGB to Linear RGB(伽马矫正)
    def linearize(c):
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4

    # Apply gamma correction to each channel
    linear_rgb = [linearize(c) for c in rgb]

    # Conversion matrix from RGB to XYZ
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    # Convert linear RGB to XYZ
    xyz = np.dot(mat, linear_rgb)

    # Extract X, Y, Z components
    X, Y, Z = xyz

    # Calculate x and y chromaticity coordinates
    denom = X + Y + Z
    if denom == 0:
        return 0, 0
    x = X / denom
    y = Y / denom

    return x, y


if __name__ == "__main__":
    # 示例RGB值，范围在[0, 1]内
    rgb = [0.5, 0.5, 0.5]

    # 调用函数并获取结果
    x, y = rgb_to_cie_xy(rgb)

    print(f"CIE色度坐标: x={x:.4f}, y={y:.4f}")
