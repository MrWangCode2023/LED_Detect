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
edges = cv2.Canny(blurred, 50, 85)

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
cv2.imshow("2", filtered_edges)
print(f"Number of detected edges: {num_edges}")

cv2.waitKey(0)
cv2.destroyAllWindows()
