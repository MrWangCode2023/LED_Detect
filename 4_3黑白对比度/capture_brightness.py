def capture_brightness(image, points):
    """
    Capture the brightness at specific points in the image.

    :param image: Input image.
    :param points: List of points (x, y) where brightness needs to be measured.
    :return: List of brightness values at the specified points.
    """
    brightness_values = []
    for (x, y) in points:
        # Ensure the point is within image bounds
        x = min(max(x, 0), image.shape[1] - 1)
        y = min(max(y, 0), image.shape[0] - 1)
        brightness_values.append(image[y, x])
    return brightness_values
