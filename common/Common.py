import cv2
import numpy as np
from collections import namedtuple
import time
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from collections import namedtuple
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt


############################### 目录 ####################################
"""
1 LED区域检测:                          object_extraction(image)
2. 获取目标区域像素值:                    object_color_extraction(image)
3. 用窗口展示像素值的颜色:                show_LED_color(color=(0, 0, 0))
4. 目标区域曲线拟合:                     object_curve_fitting(image)
5. 对图像进行自适应resize:               auto_resize(image, new_shape=(640, 640))
6. 对曲线进行等分操作:                   curve_division(curve_length, curve_coordinates, num_divisions=30)
7. 基于坐标点生成ROI：                   draw_rectangle_roi_base_on_points(image, points, roi_size=20)
                                      draw_circle_roi_base_on_points(image, points, roi_size=20)
8 输入image获取目标区域等分ROI:           draw_rectangle_roi_base_on_points(image, num_divisions=50, roi_size=20)
                                      draw_circle_roi_base_on_points(image, num_divisions=50, roi_size=20)
9 分析ROI区域像素                       analyze_image_with_rois(image, num_divisions=50, roi_size=20, brightness_threshold=50)   
"""
############################### 1. LED区域检测 ####################################
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
    cv2.rectangle(img, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)

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

    # print("Number of Objects:", roi_count)
    # cv2.imshow("Image", image)
    # cv2.imshow("Binary Image", binary)
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return filtered_contours, binary, roi_count
############################## 2. 获取目标区域像素值 ##################################
def object_color_extraction(image):
    ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
    filtered_contours, binary, object_count = object_extraction(image)
    object_color_image = cv2.bitwise_and(image, image, mask=binary)
    # 计算掩码图像中非零像素的数量
    nonzero_pixel_count = float(np.count_nonzero(binary))

    # 通道拆分
    blue_channel = object_color_image[:, :, 0]
    green_channel = object_color_image[:, :, 1]
    red_channel = object_color_image[:, :, 2]

    blue_sum = np.sum(blue_channel)
    green_sum = np.sum(green_channel)
    red_sum = np.sum(red_channel)

    # 三个通道区域的像素分别求均值
    ROI_BGR_mean = ()  # 空元组
    ROI_BGR_mean += (blue_sum / nonzero_pixel_count,)
    ROI_BGR_mean += (green_sum / nonzero_pixel_count,)
    ROI_BGR_mean += (red_sum / nonzero_pixel_count,)

    # BGR均值转换为HSV格式
    bgr_mean = np.array([[ROI_BGR_mean]], dtype=np.uint8)
    ROI_HSV_mean = cv2.cvtColor(bgr_mean, cv2.COLOR_BGR2HSV)[0][0]
    brightness = ROI_HSV_mean[2]

    color_image = show_object_color(ROI_BGR_mean)
    # cv2.imshow("binary", binary)
    # cv2.imshow("object_color_image", object_color_image)
    # cv2.imshow("color", color_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, object_count)

####################################### 3. 用窗口展示像素值的颜色 #########################################
def show_object_color(color=(0, 0, 0)):
    # 定义图像的宽度和高度
    width, height = 640, 640
    # 创建一个纯色图像，大小为 width x height，数据类型为 uint8
    color_image = np.full((height, width, 3), color, dtype=np.uint8)
    return color_image

####################################### 4. 目标区域曲线拟合 ###############################################
def object_curve_fitting(image):
    curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
    filtered_contours, binary, roi_count = object_extraction(image)
    binary_image = binary.copy()

    # 细化算法API
    skeleton_image = cv2.ximgproc.thinning(binary_image)  # 只有一个像素的线
    skeleton = skeletonize(skeleton_image // 255)  # Convert to boolean and skeletonize
    curve_image = img_as_ubyte(skeleton)

    nonzero_pixels = np.nonzero(curve_image)

    # 如果没有检测到曲线，返回None
    if len(nonzero_pixels[0]) == 0:
        return curve(None, None, None)

    curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
    curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)  # 接受的参数为数组类型

    # cv2.imshow('image', image)
    # cv2.imshow('binary', binary_image)
    # cv2.imshow('curve_img', curve_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return curve(curve_image, curve_coordinates, curve_length)


####################################### 5. 对图像进行自适应resize #########################################
def auto_resize(image, new_shape=(640, 640)):
    # 当前图像的形状 [高度, 宽度]
    shape = image.shape[:2]
    # 如果新形状是整数，则将其转换为元组
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 缩放比例 (新 / 旧)
    k = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 计算调整后的大小
    new_size = int(round(shape[1] * k)), int(round(shape[0] * k))
    # 调整图像大小
    if shape[::-1] != new_size:  # 调整大小
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return image


####################################### 6. 对曲线进行等分操作 #############################################
def curve_division(curve_length, curve_coordinates, num_divisions=50):
    # 存储等分点的坐标和角度
    points_and_angles = []

    # 等分长度
    segment_length = curve_length / num_divisions
    accumulated_length = 0.0
    coordinates_num = len(curve_coordinates)

    divided_point = curve_coordinates[0]
    points_and_angles.append((divided_point, 0))  # 初始点的角度设为0

    for i in range(1, coordinates_num):
        prev_point = curve_coordinates[i - 1]
        next_point = curve_coordinates[i]
        distance = np.linalg.norm(next_point - prev_point)
        accumulated_length += distance

        while accumulated_length >= segment_length:
            accumulated_length -= segment_length
            t = 1 - (accumulated_length / distance)
            divided_point = (1 - t) * prev_point + t * next_point
            if len(points_and_angles) == 1:
                angle = np.arctan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) * 180 / np.pi
            else:
                angle = points_and_angles[-1][1]
            points_and_angles.append((divided_point, angle))

    for i in range(1, len(points_and_angles) - 1):
        prev_point = points_and_angles[i - 1][0]
        next_point = points_and_angles[i + 1][0]
        tangent_vector = next_point - prev_point
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        angle = np.arctan2(normal_vector[1], normal_vector[0]) * 180 / np.pi
        points_and_angles[i] = (points_and_angles[i][0], angle)

    return points_and_angles  # [points, angles]

########################################### 7 基于坐标点生成ROI ###############################################
# 绘制矩形ROI
def draw_rectangle_roi_base_on_points(image, points, roi_size=20):
    """
    Draw ROI (Region of Interest) rectangles on an image centered at each point.

    Parameters:
    - image: The input image
    - points: List of coordinate points, format [(x1, y1), (x2, y2), ..., (xn, yn)]
    - roi_size: Size of the ROI rectangles (default is 20)

    Returns:
    - image_with_roi: Image with the ROI rectangles drawn
    """
    image_with_roi = image.copy()
    half_size = roi_size // 2

    for (x, y) in points:
        top_left = (x - half_size, y - half_size)
        bottom_right = (x + half_size, y + half_size)
        cv2.rectangle(image_with_roi, top_left, bottom_right, color=(255, 255, 255), thickness=2)

    return image_with_roi  # 返回绘制了ROI的图像

# 绘制圆形ROI
def draw_circle_roi_base_on_points(image, points, roi_size=20):
    """
    Draw ROI (Region of Interest) circles on an image centered at each point.

    Parameters:
    - image: The input image
    - points: List of coordinate points, format [(x1, y1), (x2, y2), ..., (xn, yn)]
    - roi_size: Diameter of the ROI circles (default is 20)

    Returns:
    - image_with_roi: Image with the ROI circles drawn
    """
    image_with_roi = image.copy()
    radius = roi_size // 2  # 半径是直径的一半

    for (x, y) in points:
        cv2.circle(image_with_roi, (x, y), radius, color=(0, 255, 0), thickness=2)

    return image_with_roi  # 返回绘制了ROI的图像


############################################# 8 输入image获取目标区域等分ROI ##############################################
def draw_rectangle_roi_base_on_divisions(image, num_divisions=50, roi_size=20):
    curve = object_curve_fitting(image)  # 获取曲线数据

    # 如果没有检测到曲线，直接返回原始图像和空的ROI列表
    if curve is None:
        return image.copy(), []

    points_and_angles = curve_division(curve.curve_length, curve.curve_coordinates, num_divisions)
    image_with_roi = image.copy()
    half_size = roi_size // 2
    rois = []  # 用于存储每个ROI的顶点坐标

    for (point, angle) in points_and_angles:
        center = (int(point[0]), int(point[1]))
        size = (roi_size, roi_size)

        # 绘制旋转矩形
        rect = (center, size, angle)
        box = cv2.boxPoints(rect)  # 计算出旋转矩形坐标顶点
        box = np.intp(box)
        cv2.drawContours(image_with_roi, [box], 0, (255, 255, 255), 2)
        rois.append(box.tolist())  # 将ROI顶点坐标添加到列表中

    return image_with_roi, rois  # 返回绘制ROI的图像和roi顶点坐标

############################################ 9 分析ROI区域像素 #####################################################
def analyze_image_with_rois(image, num_divisions=30, roi_size=20, brightness_threshold=50):
    # 调用绘制ROI的函数，获取带有绘制ROI的图像和ROI顶点坐标
    image_with_roi, rois = draw_rectangle_roi_base_on_divisions(image, num_divisions, roi_size)

    # 用于存储每个ROI的分析结果
    analysis_results = []

    # 遍历每个ROI，并为每个ROI分配一个编号
    for idx, roi in enumerate(rois):
        # 创建一个与输入图像大小相同的空白掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 将ROI顶点坐标转换为int32类型
        roi_corners = np.array(roi, dtype=np.int32)

        # 在掩码上填充ROI多边形区域，将ROI区域设置为白色
        cv2.fillPoly(mask, [roi_corners], 255)

        # 使用掩码从图像中提取ROI
        roi_image = cv2.bitwise_and(image, image, mask=mask)

        # 获取ROI区域的所有像素
        roi_pixels = roi_image[mask == 255]

        # 如果ROI区域没有像素，跳过该ROI
        if len(roi_pixels) == 0:
            continue

        # 计算亮度统计
        # 将ROI图像转换为灰度图像
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

        # 计算灰度图中ROI区域的平均亮度
        mean_brightness = round(np.mean(gray[mask == 255], dtype=float), 2)

        # 计算灰度图中ROI区域的最大亮度
        max_brightness = np.max(gray[mask == 255])

        # 计算灰度图中ROI区域的最小亮度
        min_brightness = np.min(gray[mask == 255])

        # 计算亮度低于阈值的像素比例
        low_brightness_ratio = round(np.sum(gray[mask == 255] < brightness_threshold) / len(roi_pixels), 2)

        # 计算平均颜色
        # 计算ROI区域的平均颜色（BGR）
        mean_color_bgr = np.mean(roi_pixels, axis=0).astype(int).tolist()

        # 将BGR转换为RGB
        mean_color_rgb = mean_color_bgr[::-1]

        # 计算平均颜色对应的CIE 1931 XYZ值
        mean_color_array = np.array(mean_color_rgb, dtype=np.float32)
        mean_color_xyz = rgb_to_CIE1931(mean_color_array)

        # 将结果添加到分析结果列表中
        analysis_results.append({
            'roi_id': idx + 1,  # 添加ROI编号
            'mean_brightness': mean_brightness,
            'max_brightness': max_brightness,
            'min_brightness': min_brightness,
            'ROI_color_RGB': mean_color_rgb,
            'low_brightness_ratio': low_brightness_ratio,
            'ROI_CIE1931_xyz': np.round(mean_color_xyz, 4).tolist()  # 添加CIE 1931 XYZ值
        })

    return image_with_roi, analysis_results  # 返回带有绘制ROI的图像和分析结果
##################################### 10 细化/骨架提取算法 #######################################
# 形态学骨架提取
def morphological_skeleton(binary_image):
    size = np.size(binary_image)
    skel = np.zeros(binary_image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(binary_image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_image, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary_image = eroded.copy()

        zeros = size - cv2.countNonZero(binary_image)
        if zeros == size:
            done = True

    return skel

# 距离变换骨架提取
def distance_transform_skeleton(binary_image, threshold_ratio=0.4):
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    _, skeleton_image = cv2.threshold(dist_transform, threshold_ratio * dist_transform.max(), 255, 0)
    skeleton_image = np.uint8(skeleton_image)
    return skeleton_image

# Skimage骨架提取
def skimage_skeleton(binary_image):
    skeleton = skeletonize(binary_image // 255)  # 将二值图像转换为布尔类型进行处理
    skeleton_image = img_as_ubyte(skeleton)  # 转换回 uint8 类型
    return skeleton_image

# ZhangSun细化算法
def zhang_suen_thinning(binary_image):
    # 确保二值图像为布尔形式（0 和 1）
    # 由于 Python 中 True 相当于 1，False 相当于 0，因此我们将二值图像除以 255 来转换为布尔形式。
    binary_image = binary_image // 255

    # 初始化两个标志变量，changing1 和 changing2，用于控制算法的循环
    changing1 = changing2 = 1

    # 当 changing1 或 changing2 非空时继续循环
    while changing1 or changing2:
        # 存储需要变为 0 的像素位置的列表
        changing1 = []
        # 遍历图像的每个像素，忽略图像边缘的像素（因为边缘像素无法被完全检查）
        for i in range(1, binary_image.shape[0] - 1):
            for j in range(1, binary_image.shape[1] - 1):
                # 获取当前像素 (i, j) 周围的八个像素值
                P2 = binary_image[i-1, j]
                P3 = binary_image[i-1, j+1]
                P4 = binary_image[i, j+1]
                P5 = binary_image[i+1, j+1]
                P6 = binary_image[i+1, j]
                P7 = binary_image[i+1, j-1]
                P8 = binary_image[i, j-1]
                P9 = binary_image[i-1, j-1]

                # 计算 0->1 的过渡次数，即从 P2 到 P9 的顺时针方向
                A = (P2 == 0 and P3 == 1) + (P3 == 0 and P4 == 1) + (P4 == 0 and P5 == 1) + (P5 == 0 and P6 == 1) + \
                    (P6 == 0 and P7 == 1) + (P7 == 0 and P8 == 1) + (P8 == 0 and P9 == 1) + (P9 == 0 and P2 == 1)

                # 计算邻域像素的非零值个数
                B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

                # 计算条件 m1 和 m2，用于确定是否删除当前像素
                m1 = P2 * P4 * P6
                m2 = P4 * P6 * P8

                # 如果当前像素为 1，并且满足删除条件，则将其添加到 changing1 列表中
                if (binary_image[i, j] == 1 and 2 <= B <= 6 and A == 1 and m1 == 0 and m2 == 0):
                    changing1.append((i, j))

        # 将所有在 changing1 中的像素设为 0
        for i, j in changing1:
            binary_image[i, j] = 0

        # 存储需要变为 0 的像素位置的列表
        changing2 = []
        # 第二次迭代，基本与第一次迭代相同，但条件略有不同
        for i in range(1, binary_image.shape[0] - 1):
            for j in range(1, binary_image.shape[1] - 1):
                P2 = binary_image[i-1, j]
                P3 = binary_image[i-1, j+1]
                P4 = binary_image[i, j+1]
                P5 = binary_image[i+1, j+1]
                P6 = binary_image[i+1, j]
                P7 = binary_image[i+1, j-1]
                P8 = binary_image[i, j-1]
                P9 = binary_image[i-1, j-1]

                A = (P2 == 0 and P3 == 1) + (P3 == 0 and P4 == 1) + (P4 == 0 and P5 == 1) + (P5 == 0 and P6 == 1) + \
                    (P6 == 0 and P7 == 1) + (P7 == 0 and P8 == 1) + (P8 == 0 and P9 == 1) + (P9 == 0 and P2 == 1)

                B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

                m1 = P2 * P4 * P8
                m2 = P2 * P6 * P8

                if (binary_image[i, j] == 1 and 2 <= B <= 6 and A == 1 and m1 == 0 and m2 == 0):
                    changing2.append((i, j))

        for i, j in changing2:
            binary_image[i, j] = 0

    # 将布尔值转换回 uint8 类型（0 和 255）
    return binary_image * 255

# 4
def binary_thinning(image):
    # Convert the image to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(image)

    # Apply the BinaryThinning method
    thinned_image = sitk.BinaryThinning(sitk_image)

    # Convert the result back to a NumPy array
    thinned_array = sitk.GetArrayFromImage(thinned_image)

    return thinned_array

############################# RGB转CIE1931 ###################################
def rgb_to_CIE1931(rgb):
    # 将RGB归一化到[0, 1]
    rgb_normalized = rgb / 255.0

    # 定义RGB到XYZ的转换矩阵 (D65 illuminant)
    rgb_to_xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    # 进行矩阵运算
    CIE1931_xyz = np.dot(rgb_normalized, rgb_to_xyz_matrix.T)

    return CIE1931_xyz

#################################################################################################
if __name__ == '__main__':
    # 开始计时
    start_time = time.time()
    image = cv2.imread('../Data/task1/task1_14.bmp')
    skeleton_image, skeleton_coordinates, curve_length = object_curve_fitting(image)

    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 输出运行时间
    print(f"程序运行时间: {elapsed_time:.2f} 秒")

    # 显示结果
    cv2.imshow('Thinned Image', skeleton_image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()