def generate_relative_points(image_size, margin=0.1, grid_size=5):
    """
    生成相对于图像大小的相对点和直径。
    """
    H, V = image_size
    diameter_H = int(0.1 * H)
    diameter_V = int(0.1 * V)
    diameter = min(diameter_H, diameter_V)

    points = []
    step_x = (1 - 2 * margin) / (grid_size - 1)
    step_y = (1 - 2 * margin) / (grid_size - 1)

    for i in range(grid_size):
        for j in range(grid_size):
            x = margin * V + i * step_x * V
            y = margin * H + j * step_y * H
            points.append((int(x), int(y)))

    return points, diameter