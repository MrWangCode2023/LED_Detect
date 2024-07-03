import cv2
import numpy as np
from Show import show_image

def object_extraction(image):
    img = np.copy(image)
    # 设置边框参数
    border_color = (0, 0, 0)  # 边框颜色，这里为黑色
    border_thickness = 3  # 边框厚度，单位为像素
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
        if cv2.contourArea(contour) >= 65:
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
    cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)

    # 按照从左上到右下的顺序对轮廓进行排序
    filtered_contours = sorted(contours1, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

    # 计算每个前景目标的尺寸（外接矩形的长宽）
    roi_count = len(filtered_contours)
    # print("Number of Objects:", roi_count)
    object_sizes = []
    object_positions = []
    for idx, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        object_sizes.append((w, h))
        object_positions.append((int(x + w/2), int(y + h/2)))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # for idx, (x, y) in enumerate(object_positions):
    #     print(f"Point {idx}: ({x}, {y})")

    image_dict = {
        "Original": image,
        # "Binary": binary,
        "Bboxes": img
    }
    # display_images_with_titles(image_dict)
    # result = (filtered_contours, binary, roi_count, object_sizes)

    # 确保对象位置为numpy数组
    object_positions = np.array(object_positions)

    return object_positions, image_dict

if __name__ == "__main__":
    image = cv2.imread("E:\workspace\Data\LED_data\\task4\\21.png")
    object_positions = object_extraction(image)
