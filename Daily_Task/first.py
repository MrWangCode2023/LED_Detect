import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.Common import object_curve_fitting, object_extraction


def is_continuous_and_uniform(curve_image):
    if curve_image is None:
        return False, False
    nonzero_pixels = np.column_stack(np.nonzero(curve_image))
    height, width = curve_image.shape
    is_continuous = True
    for y, x in nonzero_pixels:
        neighborhood = curve_image[max(0, y - 1):min(y + 2, height), max(0, x - 1):min(x + 2, width)]
        if np.sum(neighborhood) < 2:
            is_continuous = False
            break
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(curve_image), cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform)
    is_uniform = max_dist <= 1.0
    return is_continuous, is_uniform


if __name__ == "__main__":
    # image = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')
    image = cv2.imread('../../Data/LED_data/task1/task1_4.bmp')
    curve = object_curve_fitting(image)

    if curve.curve_coordinates is not None:
        is_continuous, is_uniform = is_continuous_and_uniform(curve.curve_image)
        print(f"Curve Image: {curve.curve_image is not None}")
        print(f"Curve Coordinates: {curve.curve_coordinates is not None}")
        print(f"Curve Length: {curve.curve_length}")
        print(f"Is Continuous: {is_continuous}")
        print(f"Is Uniform: {is_uniform}")

        plt.figure()
        plt.imshow(curve.curve_image, cmap='gray')
        plt.title("Fitted Curve")
        plt.show()

    else:
        print("No curve detected")
