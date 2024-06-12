import cv2
import cv2 as cv
import numpy as np
from common.auto_resize import auto_resize

def nothing(x):
    pass

def object_extraction(image):
    img = auto_resize(image)
    image_with_border = np.copy(img)
    # 设置边框参数
    border_color = (0, 0, 0)  # 边框颜色，这里为黑色
    border_thickness = 12  # 边框厚度，单位为像素
    # 计算边框的位置和大小
    height, width, _ = img.shape
    border_top = border_left = 0
    border_bottom = height - 1
    border_right = width - 1
    # 绘制边框
    cv2.rectangle(image_with_border, (border_left, border_top), (border_right, border_bottom), border_color,
                  border_thickness)

    gray = cv.cvtColor(image_with_border, cv2.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # 创建窗口
    cv2.namedWindow('Edge Detection')
    # 创建滑动条
    cv2.createTrackbar('Min Threshold', 'Edge Detection', 0, 255, nothing)
    cv2.createTrackbar('Max Threshold', 'Edge Detection', 0, 255, nothing)

    while True:
        # 获取滑动条位置
        min_val = cv2.getTrackbarPos('Min Threshold', 'Edge Detection')
        max_val = cv2.getTrackbarPos('Max Threshold', 'Edge Detection')

        # 应用 Canny 边缘检测
        edges = cv2.Canny(blurred, min_val, max_val)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        mask = np.zeros(image_with_border.shape[:2], dtype=np.uint8)
        filtered_contours = []
        for contour in contours:
            if cv.contourArea(contour) >= 200:
                filtered_contours.append(contour)
                cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)
        binary = mask
        roi_count = len(filtered_contours)

        # 显示图像
        cv2.imshow("gray image", gray)
        cv2.imshow("binary", binary)
        cv2.imshow("Edge Detection", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return filtered_contours, roi_count

if __name__ == "__main__":
    image_path = r"E:\workspace\Data\LED_data\task2\04.bmp"
    # image_path = r"../../Data/LED_data/task1/task1_12.bmp"
    image = cv.imread(image_path)
    object_extraction(image)
