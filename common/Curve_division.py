import cv2
import numpy as np
from common.Common import object_curve_fitting


def curve_division(curve_length, curve_coordinates, num_divisions=30):
    # 存储等分点的坐标和角度
    points_and_angles = []

    # 等分长度
    segment_length = curve_length / num_divisions
    accumulated_length = 0.0
    coordinates_num = len(curve_coordinates)

    divided_point = curve_coordinates[0]
    points_and_angles.append((divided_point, 0))  # 初始点的角度设为0

    for i in range(1, coordinates_num):
        prev_point = curve_coordinates[i - 1]
        next_point = curve_coordinates[i]
        distance = np.linalg.norm(next_point - prev_point)
        accumulated_length += distance

        while accumulated_length >= segment_length:
            accumulated_length -= segment_length
            t = 1 - (accumulated_length / distance)
            divided_point = (1 - t) * prev_point + t * next_point
            if len(points_and_angles) == 1:
                angle = np.arctan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) * 180 / np.pi
            else:
                angle = points_and_angles[-1][1]
            points_and_angles.append((divided_point, angle))

    for i in range(1, len(points_and_angles) - 1):
        prev_point = points_and_angles[i - 1][0]
        next_point = points_and_angles[i + 1][0]
        tangent_vector = next_point - prev_point
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        angle = np.arctan2(normal_vector[1], normal_vector[0]) * 180 / np.pi
        points_and_angles[i] = (points_and_angles[i][0], angle)

    return points_and_angles


def draw_divided_points(image, points_and_angles):
    # 在图像上绘制等分点和垂线角度
    for point, angle in points_and_angles:  # (divided_points[i], angle)
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), radius=1, color=(255, 255, 255), thickness=-1)

        # 计算垂线的终点
        length = 10  # 垂线的长度，可以根据需要调整
        x_end = int(x + length * np.cos(np.radians(angle)))
        y_end = int(y + length * np.sin(np.radians(angle)))

        # 绘制垂线
        cv2.line(image, (int(x), int(y)), (x_end, y_end), color=(0, 255, 0), thickness=1)
    return image


if __name__ == '__main__':
    # 读取图像并计算曲线长度及像素坐标
    image = cv2.imread("../../Data/LED_data/task1/task1_13.bmp")
    # image_path = "../../Data/LED_data/task1/task1_13.jpg"  # 替换为实际图像路径
    curve = object_curve_fitting(image)  # (curve_image, curve_coordinates, curve_length)

    # 计算等分点坐标及对应角度
    points_and_angles = curve_division(curve.curve_length, curve.fitted_contour, 50)

    print("分割点个数：", len(points_and_angles))
    print("分割点坐标和角度:", points_and_angles)

    # 画出均分后的点和垂线角度
    image_with_points_and_angles = draw_divided_points(image.copy(), points_and_angles)

    # 显示结果图像
    cv2.imshow('Divided Curve', image_with_points_and_angles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()