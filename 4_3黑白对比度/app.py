import cv2
from matplotlib import pyplot as plt
from capture_brightness import capture_brightness
from calculate_contrast import calculate_contrast
from generate_relative_points import generate_relative_points
from convert_relative_to_absolute import convert_relative_to_absolute


def contrast_detect(black_image, white_image):
    gray_black = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    gray_white = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)

    if gray_white is None or gray_black is None:
        print("Error loading images")
        return

    # Define the dimensions of the images
    H_white, V_white = gray_white.shape[1], gray_white.shape[0]
    H_black, V_black = gray_black.shape[1], gray_black.shape[0]

    # Generate relative points based on the dimensions of each image
    relative_points = generate_relative_points()

    # Convert relative points to absolute points for both images
    points_white = convert_relative_to_absolute(relative_points, H_white, V_white)
    points_black = convert_relative_to_absolute(relative_points, H_black, V_black)

    # Calculate the diameter for the points based on image dimensions
    diameter_white = int(min(0.01 * H_white, 0.01 * V_white))
    diameter_black = int(min(0.01 * H_black, 0.01 * V_black))

    # Capture brightness at each test point for both white and black images
    Lwhite = capture_brightness(gray_white, points_white)
    Ldark = capture_brightness(gray_black, points_black)

    # Calculate contrast values
    contrast_values = calculate_contrast(Lwhite, Ldark)

    # Print results
    print(f"Point总个数: {len(points_white)}\n")
    print("Point信息表：")
    for idx, (rel_point, abs_point_white, abs_point_black, Lw, Ld, contrast) in enumerate(
            zip(relative_points, points_white, points_black, Lwhite, Ldark, contrast_values), start=1):
        print(
            f"| 编号: {idx} | 相对质心坐标: {rel_point} | 白图质心坐标: {abs_point_white} | 黑图质心坐标: {abs_point_black} | Lwhite: {Lw} | Ldark: {Ld} | 对比度: {contrast:.2f}% |")

    # Plot the images and results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(gray_white, cmap='gray')
    ax[0].set_title('White Image')
    ax[1].imshow(gray_black, cmap='gray')
    ax[1].set_title('Black Image')

    for point in points_white:
        ax[0].add_patch(plt.Circle(point, diameter_white / 2, color='red', fill=False))
    for point in points_black:
        ax[1].add_patch(plt.Circle(point, diameter_black / 2, color='red', fill=False))

    plt.show()


if __name__ == "__main__":
    black_image_path = r"E:\workspace\Data\LED_data\task4\1.jpg"
    white_image_path = r"E:\workspace\Data\LED_data\task4\2.bmp"

    # Ensure the images are loaded correctly
    black_image = cv2.imread(black_image_path)
    white_image = cv2.imread(white_image_path)

    if black_image is None:
        print(f"Error loading black image from {black_image_path}")
    if white_image is None:
        print(f"Error loading white image from {white_image_path}")

    if black_image is not None and white_image is not None:
        contrast_detect(black_image, white_image)
