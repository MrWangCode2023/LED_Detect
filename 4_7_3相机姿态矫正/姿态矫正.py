import cv2
import numpy as np
from Object_extraction import object_extraction  # 假设您已经实现了这个函数

# 示例的相机内参系数矩阵 mtx
mtx = np.array([[600.0,   0.0, 320.0],
                [  0.0, 600.0, 240.0],
                [  0.0,   0.0,   1.0]])

# 示例的畸变系数矩阵 dist
dist = np.array([[ 0.1, -0.2,  0.0,  0.0,  0.0]])

# 设置圆点标定板的大小（例如，20x19的圆点阵列）
pattern_size = (20, 19)  # 行数，列数
square_size = 10.0  # 圆点之间的实际距离（单位：毫米）

# 根据标定板的点阵参数计算各个点位的世界坐标
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# 读取包含标定板的图像
img_path = 'E:\\workspace\\Data\\LED_data\\task4\\25.png'  # 替换成实际的图像路径
img = cv2.imread(img_path)

# 在图像中检测到的2D点
centers, _ = object_extraction(img)  # 假设这个函数返回centers和其他可能需要的信息
print("检测到的点的个数：", len(centers))

if len(centers) > 0:
    # 将检测到的 centers 转换为 NumPy 数组并进行形状调整
    centers = np.array(centers, dtype=np.float32)

    # 1. 估计姿态
    ret, rvec, tvec = cv2.solvePnP(objp, centers, mtx, dist)

    # 2. 计算标定板中心位置（平均圆点位置）
    board_center = np.mean(objp, axis=0)
    print(f"\n标定板中心位置（世界坐标系，单位：毫米）： [{board_center[0]:.4f}, {board_center[1]:.4f}, {board_center[2]:.4f}]")

    # 3. 将标定板中心位置转换到相机坐标系
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    # 将标定板中心位置从世界坐标系转换到相机坐标系
    board_center_cam = np.dot(R, board_center.T) + tvec.flatten()
    print(f"标定板中心位置（相机坐标系，单位：毫米）： [{board_center_cam[0]:.4f}, {board_center_cam[1]:.4f}, {board_center_cam[2]:.4f}]")

    # 4. 计算矫正后的偏移量和方向
    # 原点在相机坐标系中的位置
    camera_origin = np.array([0, 0, 0])

    # 计算标定板中心位置相对于相机原点的偏移量
    offset = board_center_cam - camera_origin
    print(f"\n相机相对于标定板中心偏移量（单位：毫米）： [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")

    # 计算偏移方向
    direction = offset / np.linalg.norm(offset)
    print(f"偏移方向向量： [{direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f}]")

    # 5. 计算欧拉角
    # 计算欧拉角
    theta_x = np.arctan2(R[2, 1], R[2, 2])
    theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    theta_z = np.arctan2(R[1, 0], R[0, 0])

    # 将弧度转换为角度
    theta_x = np.degrees(theta_x)
    theta_y = np.degrees(theta_y)
    theta_z = np.degrees(theta_z)

    print("\n相机和标定板的法线夹角（欧拉角，单位：度）：")
    print(f"X轴旋转角度： {theta_x:.4f} 度")
    print(f"Y轴旋转角度： {theta_y:.4f} 度")
    print(f"Z轴旋转角度： {theta_z:.4f} 度")

    # 6. 计算相机成像平面和标定板平面的夹角（欧拉角）
    # 标定板平面的法向量
    normal_board = R[:, 2]  # 取旋转矩阵的第三列作为标定板平面法向量

    # 相机成像平面的法向量
    inv_mtx = np.linalg.inv(mtx)
    normal_camera = np.dot(inv_mtx, [0, 0, 1])

    # 计算相机成像平面和标定板平面的夹角
    # 求解相机成像平面法向量和标定板平面法向量之间的夹角
    cos_angle = np.dot(normal_board, normal_camera)
    sin_angle = np.linalg.norm(np.cross(normal_board, normal_camera))
    angle_rad = np.arctan2(sin_angle, cos_angle)
    angle_deg = np.degrees(angle_rad)

    if angle_deg >= 90:
        angle_deg = 180 - angle_deg

        print(f"\n相机成像平面和标定板平面的夹角（欧拉角，单位：度）： {angle_deg:.4f} 度")

else:
    print("未能找到圆点标定板")
