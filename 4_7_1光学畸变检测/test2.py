import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 找到图像中的最大轮廓
def find_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour
    else:
        raise ValueError("No contours found in the image")

# 计算两条线段的交点
def calculate_intersection_point(p1, p2, p3, p4):
    def line_intersection(line1, line2):
        xdiff = (line1[0] - line1[2], line2[0] - line2[2])
        ydiff = (line1[1] - line1[3], line2[1] - line2[3])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None

        d = (det((line1[0], line1[1]), (line1[2], line1[3])), det((line2[0], line2[1]), (line2[2], line2[3])))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return int(x), int(y)

    intersection = line_intersection((p1[0], p1[1], p2[0], p2[1]), (p3[0], p3[1], p4[0], p4[1]))
    if intersection:
        if min(p1[0], p2[0]) <= intersection[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= intersection[1] <= max(p1[1], p2[1]):
            if min(p3[0], p4[0]) <= intersection[0] <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= intersection[1] <= max(p3[1], p4[1]):
                return intersection
    return None

# 计算两点之间的斜率
def calculate_slope(point1, point2):
    if point2[0] == point1[0]:
        return None
    return (point2[1] - point1[1]) / (point2[0] - point1[0])

# 计算垂直于给定斜率的斜率
def calculate_perpendicular_slope(slope):
    if slope is None:
        return 0
    if slope == 0:
        return None
    return -1 / slope

# 计算点与斜率确定的直线上的两个点
def line_from_slope_and_point(slope, point, length=1000):
    if slope is None:
        return [(point[0], point[1] - length), (point[0], point[1] + length)]
    dx = length / math.sqrt(1 + slope**2)
    dy = slope * dx
    return [(int(point[0] - dx), int(point[1] - dy)), (int(point[0] + dx), int(point[1] + dy))]

# 计算最小外接矩形及其属性
def minimum_bounding_rectangle(image, contour):
    rect_info_list = []
    combined_image = np.zeros_like(image)

    rect = cv2.minAreaRect(contour)
    box = np.intp(cv2.boxPoints(rect))

    center = np.mean(box, axis=0).astype(int)

    edge_centers = []
    edge_slopes = []

    for i in range(4):
        edge_center = ((box[i] + box[(i + 1) % 4]) // 2)
        edge_slope = calculate_slope(box[i], box[(i + 1) % 4])
        edge_centers.append(tuple(edge_center))
        edge_slopes.append(edge_slope)

    perpendicular_slopes = []
    for i in range(2):
        slope1 = edge_slopes[i]
        slope2 = edge_slopes[(i + 2) % 4]
        if slope1 is not None and slope2 is not None:
            perpendicular_slope = calculate_perpendicular_slope((slope1 + slope2) / 2)
        else:
            perpendicular_slope = 0
        perpendicular_slopes.append(perpendicular_slope)

    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)

    for edge_center in edge_centers:
        cv2.circle(image, tuple(edge_center), 5, (255, 0, 0), -1)

    cv2.drawContours(combined_image, [contour], -1, (255, 255, 255), 1)
    midlines = []
    for i in range(2):
        line = line_from_slope_and_point(perpendicular_slopes[i], edge_centers[i])
        midlines.append(line)
        cv2.line(combined_image, line[0], line[1], (0, 0, 255), 2)

    intersections = []
    intersection_info = []

    for i in range(2):
        line = midlines[i]
        for j in range(len(contour)):
            contour_point1 = contour[j][0]
            contour_point2 = contour[(j + 1) % len(contour)][0]
            intersection = calculate_intersection_point(line[0], line[1], contour_point1, contour_point2)
            if intersection is not None:
                intersections.append(intersection)
                intersection_info.append({
                    'intersection': intersection,
                    'slope': perpendicular_slopes[i]
                })
                cv2.circle(image, intersection, 5, (0, 255, 0), -1)

    perpendicular_lines = []
    for info in intersection_info:
        point = info['intersection']
        slope = info['slope']
        line = line_from_slope_and_point(slope, point)
        perpendicular_lines.append(line)
        cv2.line(image, line[0], line[1], (255, 0, 255), 2)

    new_intersections = []
    for i in range(0, len(perpendicular_lines), 2):
        if i + 1 < len(perpendicular_lines):
            line1 = perpendicular_lines[i]
            line2 = perpendicular_lines[i + 1]
            new_intersection = calculate_intersection_point(line1[0], line1[1], line2[0], line2[1])
            if new_intersection:
                new_intersections.append(new_intersection)
                cv2.circle(image, new_intersection, 5, (255, 255, 0), -1)

    rect_info = {
        'box': box,
        'box_center': center,
        'edge_centers': edge_centers,
        'edge_slopes': edge_slopes,
        'perpendicular_slopes': perpendicular_slopes,
        'contour': contour,
        'intersections': intersections,
        'intersection_info': intersection_info,
        'new_intersections': new_intersections
    }
    rect_info_list.append(rect_info)

    return rect_info_list, combined_image

# 封装显示和打印函数
def display_and_print_info(image, rect_info_list):
    # 在原图上绘制最小外接矩形
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_image, [rect_info_list[0]['box']], 0, (0, 255, 0), 2)

    # 标记最小外接矩形的顶点并添加标签
    box = rect_info_list[0]['box']
    points_labels = {'A': box[1], 'B': box[2], 'C': box[3], 'D': box[0]}
    for label, point in points_labels.items():
        cv2.circle(color_image, tuple(point), 5, (0, 0, 255), -1)
        cv2.putText(color_image, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 标记中垂线与前景边缘线的交点并添加标签
    intersections = rect_info_list[0]['intersections']
    intersection_labels = ['E', 'G', 'H', 'F']
    for label, point in zip(intersection_labels, intersections):
        cv2.circle(color_image, point, 5, (0, 0, 255), -1)
        cv2.putText(color_image, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 标记新交点并添加标签
    new_intersections = rect_info_list[0]['new_intersections']
    for i, point in enumerate(new_intersections, start=1):
        cv2.circle(color_image, point, 5, (255, 255, 0), -1)
        cv2.putText(color_image, f'New {i}', point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 显示结果
    plt.figure(figsize=(16, 12))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Original Image with Min Bounding Rect')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Foreground Contour Edges with Midlines and Intersections')

    plt.show()

    # 顺时针打印最小外接矩形的顶点坐标
    result_dict = {
        "A": tuple(box[1]),
        "B": tuple(box[2]),
        "C": tuple(box[3]),
        "D": tuple(box[0])
    }

    # 打印中垂线与前景边缘线的交点
    if len(intersections) >= 4:
        result_dict["E"] = intersections[0]
        result_dict["G"] = intersections[1]
        result_dict["H"] = intersections[2]
        result_dict["F"] = intersections[3]
    else:
        result_dict["intersections"] = "Not enough intersections found to label E, G, H, F"

    # 打印四个新交点
    if len(new_intersections) == 4:
        for i, point in enumerate(new_intersections, start=1):
            result_dict[f"New Intersection {i}"] = point
    else:
        result_dict["new_intersections"] = "Not enough new intersections found to form a rectangle"

    return result_dict

if __name__ == "__main__":
    # 读取图像
    image_path = "E:\\workspace\\Data\\LED_data\\task4\\19.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Error: Unable to read the image at {image_path}")

    # 找到轮廓
    max_contour = find_contour(image.copy())

    # 在原图上绘制最小外接矩形并获取相关信息
    rect_info_list, combined_image = minimum_bounding_rectangle(image.copy(), max_contour)

    # 显示和打印信息
    result = display_and_print_info(image, rect_info_list)

    # 打印结果字典
    for key, value in result.items():
        print(f"{key}: {value}")
