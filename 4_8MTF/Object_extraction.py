import cv2
import numpy as np
from Show import show_image
from Distance_of_points import distance_of_points
from Mid_point import mid_point

def object_extraction(image):
    # 设置边框参数
    border_color = (0, 0, 0)  # 边框颜色，这里为黑色
    border_thickness = 3  # 边框厚度，单位为像素
    # 计算边框的位置和大小
    height, width, _ = image.shape
    border_top = border_left = 0
    border_bottom = height - 1
    border_right = width - 1
    # 绘制图像外边框（黑色）
    cv2.rectangle(image, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)

    BBox_img = np.copy(image)
    MBox_img = np.copy(image)
    object_sizes = []
    object_positions = []

    # 提取边缘
    blurred = cv2.GaussianBlur(BBox_img, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # 过滤contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(BBox_img.shape[:2], dtype=np.uint8)
    mask1 = np.zeros(BBox_img.shape[:2], dtype=np.uint8)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= 5:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 闭合操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    iterations = 2
    # closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # 在此处绘制所有过滤后的轮廓
    binary = closed.copy()
    contours1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask1, contours1, -1, 255, thickness=cv2.FILLED)
    # 图像边框
    # cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)

    # 按照从左上到右下的顺序对轮廓进行排序
    filtered_contours = sorted(contours1, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

    # 计算每个前景目标的尺寸（外接矩形的长宽）
    roi_count = len(filtered_contours)
    print("Number of Objects:", roi_count)
    for idx, contour in enumerate(filtered_contours):
        # 计算BBox
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(BBox_img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 绿色矩形框，线条宽度为2

        # 计算MBox
        rect = cv2.minAreaRect(contour)
        center = rect[0]  # (x, y)中心坐标
        box = cv2.boxPoints(rect)  # 获取矩形的四个顶点
        box = np.intp(box)
        # A, B, C, D = box[0], box[1], box[2], box[3]
        # mid_ab = mid_point(A, B)
        # 绘制最小外接矩形
        cv2.drawContours(MBox_img, [box], 0, (0, 255, 0), 1)  # 绿色矩形框，线条宽度为2

        # 中心点
        object_positions.append((int(center[0]), int(center[1])))  # 使用最小外接矩形的中心坐标

    # 确保对象位置为numpy数组
    object_positions = np.array(object_positions)

    result = (filtered_contours, binary, roi_count, object_sizes, object_positions)

    image_dict = {
        "Original": image,
        "BBoxes": BBox_img,
        "MBoxes": MBox_img
    }
    show_image(image_dict)


    # return object_positions, image_dict
    return filtered_contours

if __name__ == "__main__":
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\34.png")
    filtered_contours = object_extraction(image)
