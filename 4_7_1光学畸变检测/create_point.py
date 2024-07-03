# import cv2
# import numpy as np
#
# def generate_dot_matrix_image(width, height, rows, cols, dot_radius, bg_color=(0, 0, 0), dot_color=(255, 255, 255)):
#     # 创建黑色背景图像
#     image = np.zeros((height, width, 3), dtype=np.uint8)
#     image[:] = bg_color
#
#     # 计算点之间的间距，并在图像四周留下更多边距
#     margin_x = width // 4
#     margin_y = height // 4
#     spacing_x = (width - 2 * margin_x) // (cols - 1)
#     spacing_y = (height - 2 * margin_y) // (rows - 1)
#
#     for i in range(rows):
#         for j in range(cols):
#             center_x = margin_x + j * spacing_x
#             center_y = margin_y + i * spacing_y
#             cv2.circle(image, (center_x, center_y), dot_radius, dot_color, -1)
#
#     return image
#
# def rotate_image(image, angle):
#     # 获取图像中心点
#     center = (image.shape[1] // 2, image.shape[0] // 2)
#     # 计算旋转矩阵
#     rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     # 进行仿射变换
#     rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
#     return rotated_image
#
# if __name__ == "__main__":
#     # 定义图像大小和点阵参数
#     image_width = 640
#     image_height = 480
#     rows = 7
#     cols = 7
#     dot_radius = 10  # 保持点的半径适中
#
#     # 生成点阵图
#     dot_matrix_image = generate_dot_matrix_image(image_width, image_height, rows, cols, dot_radius)
#     # cv2.imwrite('dot_matrix_image.png', dot_matrix_image)
#
#     # 旋转点阵图45度
#     rotated_image = rotate_image(dot_matrix_image, 45)
#     cv2.imwrite('E:\workspace\Data\LED_data\\task4\\13.png', rotated_image)
#
#     print("点阵图已生成并保存为'13.png'")
#     print("旋转后的点阵图已保存为'13.png'")


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
    print("Number of Objects:", roi_count)
    object_sizes = []
    object_positions = []
    for idx, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        object_sizes.append((w, h))
        object_positions.append((x, y))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for idx, (x, y) in enumerate(object_positions):
        print(f"Object {idx}: ({x}, {y})")

    cv2.imwrite('E:\workspace\Data\LED_data\\task4\\17.png', binary)
    image_dict = {
        "Original Image": image,
        "Binary": binary,
        "Bounding Boxes": img
    }
    show_image(image_dict)
    result = (filtered_contours, binary, roi_count, object_sizes)

    return object_positions

if __name__ == "__main__":
    image = cv2.imread("E:\workspace\Data\LED_data\\task4\\16.png")
    object_positions = object_extraction(image)

