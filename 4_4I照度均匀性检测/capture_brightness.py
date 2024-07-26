import cv2
import numpy as np


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