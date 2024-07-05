import cv2
import numpy as np
import glob

# 设置圆点标定板的大小（例如，4x11的圆点阵列）
pattern_size = (19, 20)
square_size = 1.0  # 圆点之间的实际距离

# 准备标定板的世界坐标
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# 用于存储检测到的角点
objpoints = []  # 3D 点
imgpoints = []  # 2D 点

# 读取标定图像
# images = glob.glob("E:\\workspace\\Data\\LED_data\\task4\\25.png")  # 替换成实际的标定图像路径
img = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\25.png")

# for fname in images:
# img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化处理
ret, thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)

# cv2.imshow("1", thresh)
# cv2.waitKey()

# 找到圆点阵列
ret, centers = cv2.findCirclesGrid(thresh, pattern_size, cv2.CALIB_CB_SYMMETRIC_GRID)

if ret:
    objpoints.append(objp)
    imgpoints.append(centers)

    # 绘制圆点
    cv2.drawChessboardCorners(img, pattern_size, centers, ret)
    cv2.imshow('Corners', img)
    cv2.waitKey(500)

cv2.destroyAllWindows()
print("圆点个数：", objpoints)

# 相机标定
if len(objpoints) > 0 and len(imgpoints) > 0:

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 打印标定结果
    print("相机内参数矩阵：\n", mtx)
    print("畸变系数：\n", dist)

    # 保存标定结果以供后续使用
    np.savez("camera_calibration.npz", mtx=mtx, dist=dist)
else:
    print("没有检测到有效的标定图像或标定板")
