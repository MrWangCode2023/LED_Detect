import cv2
import numpy as np
import matplotlib.pyplot as plt
from Mid_point import mid_point

def bbox(contours, image):
    # 创建一个与原图像相同大小的空白图像用于显示提取的目标区域
    result_img = image.copy()
    line_img1 = np.zeros_like(image)
    line_img2 = np.zeros_like(image)

    # 计算最小外接矩形和外接矩形
    for contour in contours:
        # 外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        box = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32)
        bbox_img = cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色表示外接矩形

        # 四边中点
        a, b, c, d = box[0], box[1], box[2], box[3]
        mid_ab = np.intp(mid_point(a, b))
        mid_bc = np.intp(mid_point(b, c))
        mid_cd = np.intp(mid_point(c, d))
        mid_da = np.intp(mid_point(d, a))

        mask_line1 = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_line2 = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)  # 用红色表示最小外接矩形
        cv2.line(mask_line1, pt1=mid_ab, pt2=mid_cd, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)
        cv2.line(mask_line2, pt1=mid_bc, pt2=mid_da, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)

        # line1提取区域像素
        line1_pixels_img = cv2.bitwise_and(image, image, mask=mask_line1)
        # line2提取区域像素
        line2_pixels_img = cv2.bitwise_and(image, image, mask=mask_line2)

        # 将提取的区域添加到空白图像上
        line_img1 = cv2.bitwise_or(line_img1, line1_pixels_img)
        # 将提取的区域添加到空白图像上
        line_img2 = cv2.bitwise_or(line_img2, line2_pixels_img)

    return line_img1, line_img2
    # return line1_pixels_img, line2_pixels_img

def mbbox(contours, image):
    # 创建一个与原图像相同大小的空白图像用于显示提取的目标区域
    result_img = image.copy()
    line_img1 = np.zeros_like(image)
    line_img2 = np.zeros_like(image)

    # 计算最小外接矩形和外接矩形
    for contour in contours:
        # 最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 四边中点
        a, b, c, d = box[0], box[1], box[2], box[3]
        mid_ab = np.intp(mid_point(a, b))
        mid_bc = np.intp(mid_point(b, c))
        mid_cd = np.intp(mid_point(c, d))
        mid_da = np.intp(mid_point(d, a))

        mask_line1 = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_line2 = np.zeros(image.shape[:2], dtype=np.uint8)

        # cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)  # 用红色表示最小外接矩形
        cv2.line(mask_line1, pt1=mid_ab, pt2=mid_cd, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)
        cv2.line(mask_line2, pt1=mid_bc, pt2=mid_da, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)

        # line1提取区域像素
        line1_pixels_img = cv2.bitwise_and(image, image, mask=mask_line1)
        # line2提取区域像素
        line2_pixels_img = cv2.bitwise_and(image, image, mask=mask_line2)

        # 将提取的区域添加到空白图像上
        line_img1 = cv2.bitwise_or(line_img1, line1_pixels_img)
        # 将提取的区域添加到空白图像上
        line_img2 = cv2.bitwise_or(line_img2, line2_pixels_img)

    return line_img1, line_img2
    # return line1_pixels_img, line2_pixels_img

if __name__ == "__main__":
    # 读取图像
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\34.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 找到前景区域的轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 调用函数
    b_line_img1, b_line_img2 = bbox(contours, image)
    line_img1, line_img2 = mbbox(contours, image)

    # 显示结果
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(b_line_img1, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Box')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(b_line_img2, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Box')

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(line_img1, cv2.COLOR_BGR2RGB))
    plt.title('Line 1 Extracted Pixels')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(line_img2, cv2.COLOR_BGR2RGB))
    plt.title('Line 2 Extracted Pixels')

    plt.show()
