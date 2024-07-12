import cv2
import numpy as np
from Show import show_image
# from Draw_graph import draw_graph


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

    img = np.copy(image)
    object_sizes = []
    object_positions = []

    # 提取边缘
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # 过滤contour
    contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    filtered_contours = []
    for contour in contours1:
        if cv2.contourArea(contour) >= 5:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 对过滤后的轮廓进行闭合操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    iterations = 2
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    # closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)

    # 在此处绘制所有闭合后的轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 按照从左上到右下的顺序对轮廓进行排序
    filtered_contours = sorted(contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

    result = (filtered_contours, closed, object_sizes, object_positions)


    return filtered_contours

if __name__ == "__main__":
    from Draw_graph import draw_graph
    image = cv2.imread("E:\workspace\Data\LED_data\\4_9\\1.png")
    filtered_contours = object_extraction(image)
    result = draw_graph(image, filtered_contours)
