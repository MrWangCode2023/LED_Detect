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