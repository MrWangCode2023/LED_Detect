import numpy as np
import cv2

def t6_transformCoordinatesSystem(p1, p0, points=None):
    # 将 p0 设为新的原点 (0, 0)
    origin = p0

    # 检查 points 是否为 None 或者空
    if points is None or len(points) == 0:
        return np.array([]), None, None  # 返回空数组和 None

    # 计算 x 轴方向（从 p0 到 p1 的向量）
    x_axis_vector = p1 - p0  # 方向从 p0 指向 p1
    x_axis_vector_normalized = x_axis_vector / np.linalg.norm(x_axis_vector)

    # 计算 y 轴方向（与 x 轴正方向逆时针旋转 90 度的向量）
    y_axis_vector = np.array([-x_axis_vector_normalized[1], x_axis_vector_normalized[0]])

    # 将每个点在新的坐标系中表示
    transformed_points = np.array([
        [np.dot(point - origin, x_axis_vector_normalized), np.dot(point - origin, y_axis_vector)]
        for point in points
    ])

    return transformed_points, x_axis_vector_normalized, y_axis_vector



if __name__ == "__main__":
    # image = cv2.imread('../../projectData/LED_data/task4/2.bmp')
    image = cv2.imread('../../projectData/LED_data/task4/6.jpg')

    # 示例坐标点
    p1 = np.array([196, 217])  # 原坐标系中的点（在 x 轴负半轴上）
    p0 = np.array([204, 170])  # 新坐标系的原点
    points = np.array([[196, 217], [204, 171], [261, 160], [262, 190]])

    # 转换坐标
    transformed_points, x_axis_vector_normalized, y_axis_vector = t6_transformCoordinatesSystem(p1, p0, points)

    # 打印转换后的坐标
    print("Transformed Points:\n", transformed_points)

    # 创建一张空白图像以进行可视化
    image_height, image_width = 500, 500
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # 绘制原始坐标点
    for point in points:
        cv2.circle(image, tuple(point), 5, (255, 0, 0), -1)  # 蓝色

    # 绘制新原点 (P2)
    cv2.circle(image, tuple(p0), 5, (0, 0, 255), -1)  # 红色

    # 绘制原始坐标系
    cv2.line(image, tuple(p1), tuple(p0), (0, 255, 0), 2)  # 绿色

    # 绘制新的坐标系
    new_x_end = (p0 + x_axis_vector_normalized * 50).astype(int)
    new_y_end = (p0 + y_axis_vector * 50).astype(int)
    cv2.line(image, tuple(p0), tuple(new_x_end), (0, 165, 255), 2)  # 橙色
    cv2.line(image, tuple(p0), tuple(new_y_end), (128, 0, 128), 2)  # 紫色

    # 显示图像
    cv2.imshow('Transformed Coordinate System', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
