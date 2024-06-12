import cv2
import numpy as np

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
    cv2.rectangle(img, (border_left, border_top), (border_right, border_bottom), border_color,
                  border_thickness)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    edges = cv2.Canny(blurred, threshold1=120, threshold2=240)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= 800:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iterations = 3
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # 在此处绘制所有过滤后的轮廓
    binary = closed.copy()
    cv2.drawContours(binary, filtered_contours, -1, 255, thickness=cv2.FILLED)
    cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color,
                  border_thickness)

    roi_count = len(filtered_contours)

    print("Number of Objects:", roi_count)
    cv2.imshow("Image", image)
    cv2.imshow("Binary Image", binary)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered_contours, binary, roi_count

if __name__ == "__main__":
    # 示例图像路径
    image_path = "../../Data/LED_data/task1/task1_6.bmp"  # 替换为实际图像路径

    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or unable to read")
        exit()
    contours, binary, ROI_count = object_extraction(image)

