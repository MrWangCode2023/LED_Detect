import cv2
import numpy as np

# 读取图像
image = cv2.imread("E:\\workspace\\Data\\LED_data\\Light_data\\1.jpg", cv2.IMREAD_COLOR)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 直方图均衡化，提高对比度
equalized = cv2.equalizeHist(gray)

# 高斯模糊，用于去除噪声
blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

# Canny边缘检测，降低阈值
edges = cv2.Canny(blurred, 30, 100)

# 应用Laplacian算子
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 结合Canny和Laplacian结果
combined_edges = cv2.bitwise_or(edges, laplacian)

# 显示图像
cv2.imshow("Original Image", image)
cv2.imshow("Equalized Image", equalized)
cv2.imshow("Edges", edges)
cv2.imshow("Laplacian", laplacian)
cv2.imshow("Combined Edges", combined_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
