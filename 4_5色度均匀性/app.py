import cv2
import numpy as np
from matplotlib import pyplot as plt
from generate_relative_points import generate_relative_points


def capture_and_convert_colors(image, points, diameter):
    """
    从图像中捕获指定点的平均颜色，并转换为CIE xy色坐标。
    """
    def rgb_to_cie_xy(rgb):
        rgb = np.array(rgb) / 255.0

        def linearize(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        linear_rgb = np.array([linearize(c) for c in rgb])

        mat = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])

        xyz = np.dot(mat, linear_rgb)
        X, Y, Z = xyz
        denom = X + Y + Z
        if denom == 0:
            return 0, 0
        return X / denom, Y / denom

    color_coordinates = []
    radius = diameter // 2

    for (x, y) in points:
        if x < radius or y < radius or x >= image.shape[1] - radius or y >= image.shape[0] - radius:
            print(f"Skipping point ({x}, {y}) due to size mismatch.")
            continue

        circular_area = image[y - radius:y + radius, x - radius:x + radius]

        if circular_area.shape != (diameter, diameter, 3):
            continue

        avg_color = np.mean(circular_area, axis=(0, 1))

        if np.isnan(avg_color).any() or len(avg_color) != 3:
            continue

        color_coordinates.append(rgb_to_cie_xy(avg_color))

    return color_coordinates


def calculate_uniformity(xy_coordinates):
    """
    计算色坐标的均匀性。
    """
    Cxavg = np.mean([coord[0] for coord in xy_coordinates])
    Cyavg = np.mean([coord[1] for coord in xy_coordinates])
    Ax = max([abs(coord[0] - Cxavg) for coord in xy_coordinates])
    Ay = max([abs(coord[1] - Cyavg) for coord in xy_coordinates])
    return Cxavg, Cyavg, Ax, Ay


def brightness_uniformity_detect(image_path):
    """
    检测图像的亮度均匀性。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image from {image_path}")
        return

    H, V, _ = image.shape
    points, diameter = generate_relative_points((H, V))

    xy_coordinates = capture_and_convert_colors(image, points, diameter)

    if not xy_coordinates:
        print("No valid color coordinates captured.")
        return

    Cxavg, Cyavg, Ax, Ay = calculate_uniformity(xy_coordinates)

    print(f"整体25个点的x色坐标平均值: {Cxavg:.4f}")
    print(f"整体25个点的y色坐标平均值: {Cyavg:.4f}")
    print(f"x色坐标均匀性: {Ax:.4f}")
    print(f"y色坐标均匀性: {Ay:.4f}")

    print("\n各点相对色坐标和均匀性:")
    for idx, ((x, y), (Cxm, Cym)) in enumerate(zip(points, xy_coordinates), start=1):
        Ax_point = abs(Cxm - Cxavg)
        Ay_point = abs(Cym - Cyavg)
        print(f"| Point {idx} | 位置 (相对于图像大小) ({x}, {y}) | 色坐标 (Cx: {Cxm:.4f}, Cy: {Cym:.4f}) | 均匀性 (Ax: {Ax_point:.4f}, Ay: {Ay_point:.4f}) |")

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for (x, y) in points:
        plt.plot(x, y, 'ro', markersize=min(0.1 * V, 0.1 * H))  # 标记点的大小根据图像计算
    plt.title('Test with Marked Points')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    white_image_path = r"E:\workspace\Data\LED_data\task4\1.bmp"
    brightness_uniformity_detect(white_image_path)
