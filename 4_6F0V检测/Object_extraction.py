import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from Show import display_images_with_titles

def object_extraction(image):
    img = np.copy(image)
    # 设置边框参数
    border_color = (0, 0, 0)  # 边框颜色，这里为黑色
    border_thickness = 7  # 边框厚度，单位为像素
    # 计算边框的位置和大小
    height, width, _ = img.shape
    border_top = border_left = 0
    border_bottom = height - 1
    border_right = width - 1
    # 绘制边框
    cv2.rectangle(img, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)

    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    edges = cv2.Canny(blurred, threshold1=80, threshold2=240)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask1 = np.zeros(img.shape[:2], dtype=np.uint8)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= 10:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 闭合操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    iterations = 10
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # 在此处绘制所有过滤后的轮廓
    binary = closed.copy()
    contours1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask1, contours1, -1, 255, thickness=cv2.FILLED)
    cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)

    # 计算每个前景目标的尺寸（外接矩形的长宽）
    roi_count = len(filtered_contours)
    print("Number of Objects:", roi_count)
    object_sizes = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        object_sizes.append((w, h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # print("Object sizes (width, height):", object_sizes)
    # print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    # print(f"Closed shape: {closed.shape}, dtype: {closed.dtype}")

    image_dict = {
        "Original Image": image,
        # "Edges": edges,
        # "Mask": mask,
        # "Mask1": mask1,
        # "Binary Image": closed,
        "Bounding Boxes": img
    }
    display_images_with_titles(image_dict)
    result = (filtered_contours, binary, roi_count, object_sizes)

    return object_sizes