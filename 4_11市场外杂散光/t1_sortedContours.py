import cv2
import numpy as np
from PIL import Image
from Show import show_image


def t1_sortedContours(pred_img):
    filted_contours = []

    # 3 边缘检测
    edges = cv2.Canny(pred_img, 50, 150)

    # 4 查找轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5 轮廓过滤
    """
    1.面积过滤：根据轮廓的面积过滤轮廓，可以去除过小或过大的轮廓。
    2.轮廓长度（周长）过滤：根据轮廓的周长过滤，可以去除异常长或短的轮廓。
    3. 形状过滤：通过计算轮廓的形状特征（如圆度、长宽比等）来过滤轮廓。
    4. 轮廓凸性过滤：根据轮廓的凸性过滤，即轮廓的凸性度量（如轮廓的凸包与原轮廓的面积比）。
    5. 轮廓凸包过滤: 通过计算轮廓的凸包（convex hull）来过滤轮廓，保留那些凸包与轮廓本身相近的轮廓。
    6. 轮廓位置过滤: 根据轮廓的位置（如距离图像边缘的距离）过滤轮廓。
    7. 轮廓角点数过滤: 通过计算轮廓的角点数（如多边形近似）来过滤轮廓。
    """
    min_area, max_area, min_perimeter, max_perimeter = 50, 1000, 50, 1000
    # max_area1, max_area1_index = 0, -1
    for idx, cnt in enumerate(contours):
        ## 5.1 面积+周长过滤
        area , perimeter = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
        if min_area < area and min_perimeter < perimeter:
            filted_contours.append(cnt)
            # if area > max_area1:
            #     max_area1 = area
            #     max_area_contour = cnt
            #     max_area_index = idx

    # 6 根据轮廓面积进行排序
    contours_sorted = sorted(filted_contours, key=cv2.contourArea, reverse=True)

    return contours_sorted


if __name__ == "__main__":
    # E:\workspace\projectData\LED_data\4_11
    # image = cv2.imread(r"E:\workspace\Data\LED_data\\task1\\6.bmp")
    image = cv2.imread(r"../../../projectData/LED_data/4_11/3.bmp")
    image1 = image.copy()

    # 1 添加边框
    top, bottom, left, right = [1, 1, 1, 1]  # 边框大小
    color = [0, 0, 0]  # 黑色
    img2 = cv2.copyMakeBorder(image1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # 2 图像预处理
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
    blured = cv2.GaussianBlur(closed, (5, 5), 0)

    contours_sorted = t1_sortedContours(blured)
    print(f"检测到的轮廓数量：{len(contours_sorted)}")
    cv2.drawContours(image1, contours_sorted, -1, (0, 255, 0), 1)

    cv2.imshow('Origin', image)
    cv2.imshow('contours_sorted', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()