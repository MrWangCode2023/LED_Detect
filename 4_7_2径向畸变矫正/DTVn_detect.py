import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from Show import show_image
from Edge_center_to_vertext import edge_center_to_vertext

def find_max_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    else:
        raise ValueError("No contours found in the image")

def calculate_slope(point1, point2):
    if point2[0] == point1[0]:
        return None
    return (point2[1] - point1[1]) / (point2[0] - point1[0])

def calculate_perpendicular_slope(slope):
    if slope is None:
        return 0
    if slope == 0:
        return None
    return -1 / slope

def line_from_slope_and_point(slope, point, length=1000):
    if slope is None:
        return [(point[0], point[1] - length), (point[0], point[1] + length)]
    dx = length / math.sqrt(1 + slope**2)
    dy = slope * dx
    return [(int(point[0] - dx), int(point[1] - dy)), (int(point[0] + dx), int(point[1] + dy))]

def calculate_intersection_point(p1, p2, p3, p4):
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])
    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(p1, p2), det(p3, p4))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    intersection = (int(x), int(y))

    if min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]):
        if min(p3[0], p4[0]) <= x <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= y <= max(p3[1], p4[1]):
            return intersection
    return None

# def

def dtv_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    origin_image = image.copy()
    # result_image = np.zeros_like(image)
    result_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255

    # 计算前景最大轮廓
    max_contour = find_max_contour(gray)
    # 绘制前景最大轮廓边缘
    cv2.drawContours(result_image, [max_contour], -1, (0, 0, 0), 1)

    # 最小外接矩形
    rect = cv2.minAreaRect(max_contour)
    box = np.intp(cv2.boxPoints(rect))
    center = tuple(np.mean(box, axis=0).astype(int))  # 矩形中心
    edge_centers = [tuple((box[i] + box[(i + 1) % 4]) // 2) for i in range(4)]  # 最小外接矩形四边中心
    edge_slopes = [calculate_slope(box[i], box[(i + 1) % 4]) for i in range(4)]  # 角度
    perpendicular_slopes = [calculate_perpendicular_slope((edge_slopes[i] + edge_slopes[(i + 2) % 4]) / 2) for i in range(2)]
    # 最小外接矩形框顶点
    A , B, C, D, O = box[1], box[2], box[3], box[0], center
    rectangel_ABCD = {'A': A,
                     'B': B,
                     'C': C,
                     'D': D,
                     }
    min_rectangle = {'A': A,
                     'B': B,
                     'C': C,
                     'D': D,
                     "O": O
                     }
    print("最小外接矩形坐标点：")
    for label, point in rectangel_ABCD.items():
        cv2.circle(result_image, point, 5, (0, 0, 0), -1)
        cv2.putText(result_image, label,
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2,
                    cv2.LINE_AA)
        print(f"{label}: {point}")

    # 最小外接就行四边中心
    E, F, G, H = edge_centers[0], edge_centers[1], edge_centers[2], edge_centers[3]
    rectangle_edge_centers = {
        'E': E,
        'F': F,
        'G': G,
        'H': H
    }
    # print("\n最小外接矩形四边中点坐标点：")
    # for label, point in rectangle_edge_centers.items():
    #     cv2.circle(result_image, point, 5, (0, 0, 0), -1)
    #     cv2.putText(result_image, label,
    #                 (point[0] + 5, point[1] - 5),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.8, (0, 0, 255), 2,
    #                 cv2.LINE_AA)
    #     print(f"{label}: {point}")


    cv2.drawContours(result_image, [box], 0, (0, 0, 0), 1)  # 绘制最小外接矩形

    # 中垂线
    midlines = [line_from_slope_and_point(perpendicular_slopes[i], edge_centers[i]) for i in range(2)]

    # 得到轮廓边缘中心交点坐标
    center_intersection_points = []
    for i, line in enumerate(midlines):
        for j in range(len(max_contour)):
            contour_point1 = max_contour[j][0]
            contour_point2 = max_contour[(j + 1) % len(max_contour)][0]
            intersection = calculate_intersection_point(line[0], line[1], contour_point1, contour_point2)
            if intersection is not None:
                center_intersection_points.append(intersection)
    # 轮廓边缘中点
    E1, F1, G1, H1 = center_intersection_points[0], center_intersection_points[3], \
        center_intersection_points[1], center_intersection_points[2]
    Center_intersection_points = {"E1": E1,
                                  "G1": G1,
                                  "H1": H1,
                                  "F1": F1}
    # print("\n内接矩形四边中心坐标：")
    # for label, point in Center_intersection_points.items():
    #     cv2.circle(result_image, point, 5, (0, 0, 0), -1)
    #     cv2.putText(result_image, label,
    #                 (point[0] + 5, point[1] - 5),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.8, (255, 0, 0), 2,
    #                 cv2.LINE_AA)
    #     print(f"{label}: {point}")

    # 基于内接矩形四边中心坐标计算内接矩形顶点坐标
    A1, B1, C1, D1 = edge_center_to_vertext(E1, F1, G1, H1, image)
    rectangel_A1B1C1D1 = {
        "A1": A1,
        "B1": B1,
        "C1": C1,
        "D1": D1,
    }
    print("\n内接矩形顶点坐标：")
    for label, point in rectangel_A1B1C1D1.items():
        cv2.circle(result_image, point, 5, (0, 0, 0), -1)
        cv2.putText(result_image, label,
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2,
                    cv2.LINE_AA)
        print(f"{label}: {point}")

    # 绘制矩形
    pts = np.array([A1, B1, C1, D1], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(result_image, [pts], isClosed=True, color=(0, 0, 0), thickness=1)

    image_dict = {
        # "Origin_image": origin_image,
        "Result_image": result_image
    }
    # show_image(image_dict)

    # 计算DTV
    x1, y1 = E[0], E[1]
    x2, y2 = F[0], F[1]
    x3, y3 = G[0], G[1]
    x4, y4 = H[0], H[1]
    xa, ya = A[0], A[1]
    xb, yb = B[0], B[1]
    xd, yd = D[0], D[1]

    x11, y11 = E1[0], E1[1]
    x21, y21 = F1[0], F1[1]
    x31, y31 = G1[0], G1[1]
    x41, y41 = H1[0], H1[1]

    d_EE1 = math.sqrt((y1 - y11) ** 2 + (x1 - x11) ** 2)
    d_FF1 = math.sqrt((y2 - y21) ** 2 + (x2 - x21) ** 2)
    d_GG1 = math.sqrt((y3 - y31) ** 2 + (x3 - x31) ** 2)
    d_HH1 = math.sqrt((y4 - y41) ** 2 + (x4 - x41) ** 2)

    d_AB = math.sqrt((ya - yb) ** 2 + (xa - xb) ** 2)
    d_AD = math.sqrt((ya - yd) ** 2 + (xa - xd) ** 2)

    dtv_e = d_EE1 / d_AB
    dtv_f = d_FF1 / d_AD
    dtv_g = d_GG1 / d_AB
    dtv_h = d_HH1 / d_AD

    DTVn = [dtv_e, dtv_f, dtv_g, dtv_h]
    max_DTV = 0
    for dtv in DTVn:
        max_DTV = max(max_DTV, dtv)

    # print("\nDTVn:", DTVn)
    # print("MaxDTV:", max_DTV)
    DTV_dict = {
        "DTVn": DTVn,
        "max_DTV": max_DTV
    }

    return image_dict, DTV_dict


if __name__ == "__main__":
    image_path = "E:\\workspace\\Data\\LED_data\\task4\\19.png"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to read the image at {image_path}")
    color_image, result_image = dtv_detect(image.copy())
