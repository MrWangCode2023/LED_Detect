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