import cv2
import numpy as np

# 读取图像
image = cv2.imread("E:\\workspace\\Data\\LED_data\\Light_data\\3.bmp", cv2.IMREAD_COLOR)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 直方图均衡化，提高对比度
equalized = cv2.equalizeHist(gray)

# 高斯模糊，用于去除噪声
blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

# Canny边缘检测，降低阈值(30, 100)
edges = cv2.Canny(blurred, 10, 30)

# 应用Laplacian算子
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 结合Canny和Laplacian结果
combined_edges = cv2.bitwise_or(edges, laplacian)

# 过滤掉小的边缘
contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_edges = np.zeros_like(combined_edges)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:  # 过滤掉面积小于100的轮廓
        cv2.drawContours(filtered_edges, [contour], -1, 255, 1)

# 计算检测到的边缘数量
num_edges = np.sum(combined_edges > 0)


# 显示图像
cv2.imshow("Original Image", image)
cv2.imshow("Combined Edges", combined_edges)

print(f"Number of detected edges: {num_edges}")

cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# # 读取图像
# # image = cv2.imread("E:\\workspace\\Data\\LED_data\\Light_data\\3.bmp", cv2.IMREAD_COLOR)
# str_img_file = "E:\\workspace\\Data\\LED_data\\Light_data\\3.bmp"
# m_src = cv2.imread(str_img_file)
#
# if m_src is None:
#     raise ValueError("Image not found or unable to open")
#
# m_src2 = m_src.copy()
#
# # 转换为灰度图像
# m_gray = cv2.cvtColor(m_src, cv2.COLOR_BGR2GRAY)
#
# # 高斯模糊
# m_gray = cv2.GaussianBlur(m_gray, (5, 5), 1.0)
# m_gray2 = m_gray.copy()
#
# cv2.imshow("gray", m_gray)
#
# # 方法1：利用梯度变化检测缺陷
# m_sobel_x = cv2.Sobel(m_gray, cv2.CV_16S, 1, 0, ksize=7)
# m_sobel_y = cv2.Sobel(m_gray, cv2.CV_16S, 0, 1, ksize=7)
# m_sobel_x = cv2.convertScaleAbs(m_sobel_x)
# m_sobel_y = cv2.convertScaleAbs(m_sobel_y)
#
# m_edge = cv2.addWeighted(m_sobel_x, 1, m_sobel_y, 1, 0)
# cv2.imshow("edge", m_edge)
#
# _, m_thresh = cv2.threshold(m_edge, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow("thresh", m_thresh)
#
# kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
# m_morph = cv2.morphologyEx(m_thresh, cv2.MORPH_ERODE, kernel1)
# cv2.imshow("erode", m_morph)
#
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# m_morph = cv2.morphologyEx(m_morph, cv2.MORPH_DILATE, kernel2)
# cv2.imshow("dilate", m_morph)
#
# contours, _ = cv2.findContours(m_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > 200:
#         cv2.drawContours(m_src, [contour], -1, (0, 0, 255), 1)
#
# cv2.imshow("result1", m_src)
#
# # 方法2：利用局部直方图均衡化方法检测缺陷
# clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(30, 30))
# m_gray2 = clahe.apply(m_gray2)
# cv2.imshow("equalizeHist", m_gray2)
#
# _, m_thresh2 = cv2.threshold(m_gray2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# cv2.imshow("thresh2", m_thresh2)
#
# kernel2_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
# m_morph2 = cv2.morphologyEx(m_thresh2, cv2.MORPH_ERODE, kernel2_1)
# cv2.imshow("morph2", m_morph2)
#
# contours2, _ = cv2.findContours(m_morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# for contour in contours2:
#     area = cv2.contourArea(contour)
#     if area > 200:
#         cv2.drawContours(m_src2, [contour], -1, (0, 0, 255), 1)
#
# cv2.imshow("result2", m_src2)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
