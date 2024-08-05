import cv2
import numpy as np
from PIL import Image
from Show import show_image


def t1Contours(image):
    img1 = image.copy()
    filted_contours = []

    # 1 添加边框
    top, bottom, left, right = [1, 1, 1, 1]  # 边框大小
    color = [0, 0, 0]  # 黑色
    img2 = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # 2 图像预处理
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3 边缘检测
    edges = cv2.Canny(blured, 50, 150)

    # 2开操作
    # kernel_size = 3
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 创建一个内核
    # opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)  # 执行开操作

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
    for cnt in contours:
        ## 5.1 面积+周长过滤
        min_area,max_area, min_perimeter, max_perimeter = 1000, 1000, 900, 1000
        if min_area < cv2.contourArea(cnt) and min_perimeter < cv2.arcLength(cnt, True):
            filted_contours.append(cnt)

    return filted_contours


if __name__ == "__main__":
    image = cv2.imread(r"E:\workspace\Data\LED_data\\task1\\6.bmp")
    image1 = image.copy()
    filted_contours = t1Contours(image)
    print(f"检测到的轮廓数量：{len( filted_contours)}")
    cv2.drawContours(image1, filted_contours, -1, (0, 255, 0), 1)

    cv2.imshow('Origin', image)
    cv2.imshow('FiltedContours', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()