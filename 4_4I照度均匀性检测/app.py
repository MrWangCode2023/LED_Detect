import cv2
import numpy as np
from matplotlib import pyplot as plt


def capture_brightness(image, points, diameter):
    """
    Capture the brightness at specific points in the image by averaging the brightness in a circular area.

    :param image: Input image.
    :param points: List of points (x, y) where brightness needs to be measured.
    :param diameter: Diameter of the circular area to average the brightness.
    :return: List of brightness values at the specified points.
    """
    radius = diameter // 2
    brightness_values = []

    for (x, y) in points:
        x = min(max(x, radius), image.shape[1] - radius - 1)
        y = min(max(y, radius), image.shape[0] - radius - 1)

        # Extract the circular area
        circular_area = image[y - radius:y + radius + 1, x - radius:x + radius + 1]

        # Ensure the extracted area is of the correct size
        if circular_area.shape[0] != diameter or circular_area.shape[1] != diameter:
            print(
                f"Skipping point ({x}, {y}) due to size mismatch: extracted size = {circular_area.shape}, expected size = ({diameter}, {diameter}).")
            continue

        # Create a circular mask
        mask = np.zeros((diameter, diameter), dtype=np.uint8)
        cv2.circle(mask, (radius, radius), radius, 1, -1)

        # Apply the mask to the circular area
        masked_area = circular_area * mask

        # Calculate the average brightness
        avg_brightness = np.sum(masked_area) / np.sum(mask)
        brightness_values.append(avg_brightness)

    return brightness_values


def calculate_lnu(L, Lcenter):
    """
    Calculate the brightness uniformity (LNU) for a single point.

    :param L: Brightness of the test point.
    :param Lcenter: Brightness of the center test point.
    :return: Brightness uniformity (LNU) as a percentage.
    """
    if Lcenter == 0:
        return np.inf
    return (L / Lcenter) * 100


def generate_relative_points(margin=0.1, grid_size=5):
    """
    Generate relative points based on the image dimensions H and V.

    :param margin: Margin as a fraction of image dimensions (default is 0.1 for 10% margin).
    :param grid_size: The size of the grid (default is 5 for 5x5 grid).
    :return: List of points (x, y) in relative positions.
    """
    points = []
    step_x = (1 - 2 * margin) / (grid_size - 1)
    step_y = (1 - 2 * margin) / (grid_size - 1)

    for i in range(grid_size):
        for j in range(grid_size):
            x = margin + i * step_x
            y = margin + j * step_y
            points.append((round(x, 2), round(y, 2)))  # 保留两位小数

    return points


def convert_relative_to_absolute(points, width, height):
    """
    Convert relative points to absolute points based on image dimensions.

    :param points: List of relative points (x, y).
    :param width: Width of the image.
    :param height: Height of the image.
    :return: List of absolute points (x, y).
    """
    absolute_points = [(int(x * width), int(y * height)) for x, y in points]
    return absolute_points


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
            f"| 编号: {idx} | 相对质心坐标: {rel_point} | 绝对质心坐标: {abs_point} | 亮度: {L:.2f} | 亮度均匀性: {LNU:.2f}% |")

    # Calculate overall brightness uniformity (LNU)
    overall_LNU = calculate_lnu(Lavg, Lcenter)
    print(f"\nPoint总个数: {len(points)}\n整体亮度均值 (Lavg): {Lavg:.2f}\n整体亮度均匀性 (LNU): {overall_LNU:.2f}%")

    # Plot the image and results
    plt.imshow(white_image, cmap='gray')
    plt.title('White Image')

    for point in points:
        plt.gca().add_patch(plt.Circle(point, diameter / 2, color='red', fill=False))

    plt.show()


if __name__ == "__main__":
    white_image_path = r"E:\workspace\Data\LED_data\task4\2.bmp"
    brightness_uniformity_detect(white_image_path)
