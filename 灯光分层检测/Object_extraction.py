import cv2
import numpy as np

# 读取图像
image = cv2.imread("E:\\workspace\\Data\\LED_data\\Light_data\\2.bmp", cv2.IMREAD_COLOR)
image1 = image.copy()

# 转换为灰度图像
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# 高斯模糊，用于去除噪声
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# 直方图均衡化，提高对比度
# equalized = cv2.equalizeHist(blurred)



# Canny边缘检测，降低阈值
edges = cv2.Canny(blurred, 10, 30)

# 闭合操作
# 创建一个结构元素，这里使用3x3的矩形
kernel = np.ones((3, 3), np.uint8)

# 执行形态学闭操作
# MORPH_CLOSE 是用于闭操作的标志
closing1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
closing2 = cv2.morphologyEx(closing1, cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(closing2, cv2.MORPH_CLOSE, kernel)



contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filted_contours = []
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
    min_area,max_area, min_perimeter, max_perimeter = 20, 1000, 20, 1000
    if min_area < cv2.contourArea(cnt) and min_perimeter < cv2.arcLength(cnt, True):
        filted_contours.append(cnt)


# cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
cv2.drawContours(image1, filted_contours, -1, (0, 255, 0), 1)




# 显示图像
cv2.imshow("Original Image", image)
cv2.imshow("Original Image1", image1)
# cv2.imshow("Equalized Image", equalized)
# cv2.imshow("Edges", edges)
# cv2.imshow("Laplacian", laplacian)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
