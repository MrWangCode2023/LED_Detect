import cv2
import numpy as np
from matplotlib import pyplot as plt
from capture_brightness import capture_brightness
from calculate_lnu import calculate_lnu
from generate_relative_points import generate_relative_points
from convert_relative_to_absolute import convert_relative_to_absolute


def brightness_uniformity_detect(white_image_path):
    # Load the white image
    white_image = cv2.imread(white_image_path, cv2.IMREAD_GRAYSCALE)

    if white_image is None:
        print(f"Error loading white image from {white_image_path}")
        return

    H, V = white_image.shape[1], white_image.shape[0]

    # Generate relative points based on the dimensions of the image
    relative_points = generate_relative_points()

    # Convert relative points to absolute points
    points = convert_relative_to_absolute(relative_points, H, V)

    # Calculate the diameter for the points based on image dimensions
    diameter = int(min(0.1 * H, 0.1 * V))

    # Ensure the diameter is odd
    if diameter % 2 == 0:
        diameter += 1

    # Capture brightness at each test point
    brightness_values = capture_brightness(white_image, points, diameter)

    if not brightness_values:
        print("No brightness values were captured.")
        return

    # Calculate center brightness
    Lcenter = brightness_values[len(brightness_values) // 2]  # Assuming the center point is the middle one

    # Calculate brightness uniformity (LNU) for each point
    LNU_values = [calculate_lnu(L, Lcenter) for L in brightness_values]

    # Calculate overall brightness average (Lavg)
    Lavg = np.mean(brightness_values)

    # Print results
    # print(f"Point总个数: {len(points)}\n")
    print("Point信息表：")
    for idx, (rel_point, abs_point, L, LNU) in enumerate(zip(relative_points, points, brightness_values, LNU_values),
                                                         start=1):
        print(
            f"| 编号: {idx} | 相对质心坐标: {rel_point} | 绝对质心坐标: {abs_point} | 照度: {L:.2f} | 照度均匀性: {LNU:.2f}% |")

    # Calculate overall brightness uniformity (LNU)
    overall_LNU = calculate_lnu(Lavg, Lcenter)
    print(f"\nPoint总个数: {len(points)}\n整体照度均值 (Lavg): {Lavg:.2f}\n整体照度均匀性 (LNU): {overall_LNU:.2f}%")

    # Plot the image and results
    plt.imshow(white_image, cmap='gray')
    plt.title('White Image')

    for point in points:
        plt.gca().add_patch(plt.Circle(point, diameter / 2, color='red', fill=False))

    plt.show()


if __name__ == "__main__":
    white_image_path = r"E:\workspace\Data\LED_data\task4\2.bmp"
    brightness_uniformity_detect(white_image_path)
