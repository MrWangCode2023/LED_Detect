import cv2
import numpy as np
from collections import namedtuple
import time


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
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 应用二值化
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # 找到轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 过滤轮廓，假设我们只保留面积大于100的轮廓
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    binary = mask

    ROI_count = len(filtered_contours)

    # print("检测到的LED区域数量：", ROI_count)
    return filtered_contours, binary, ROI_count

############################## 2. 获取目标区域像素值 ##################################
def object_color_extraction(image):
    ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
    filtered_contours, binary, ROI_count = object_extraction(image)
    roi_color_image = cv2.bitwise_and(image, image, mask=binary)
    # 计算掩码图像中非零像素的数量
    nonzero_pixel_count = float(np.count_nonzero(binary))
    # 避免被除数为零
    if nonzero_pixel_count == 0:
        return ROI((0, 0, 0), (0, 0, 0), 0, ROI_count)

    # 通道拆分
    blue_channel = roi_color_image[:, :, 0]
    green_channel = roi_color_image[:, :, 1]
    red_channel = roi_color_image[:, :, 2]

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


    return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count)

####################################### 3. 用窗口展示像素值的颜色 #########################################
def show_LED_color(color=(0, 0, 0)):
    # 定义图像的宽度和高度
    width, height = 640, 640
    # 创建一个纯色图像，大小为 width x height，数据类型为 uint8
    color_image = np.full((height, width, 3), color, dtype=np.uint8)
    return color_image

####################################### 4. 目标区域曲线拟合 ###############################################
def object_curve_fitting(image):
    curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
    filtered_contours, binary, area_count = object_extraction(image)
    binary_image = binary.copy()
    # 细化算法API
    curve_image = cv2.ximgproc.thinning(binary_image)  # 只有一个像素的线
    nonzero_pixels = np.nonzero(curve_image)

    # 如果没有检测到曲线，返回None
    if len(nonzero_pixels[0]) == 0:
        return None

    curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
    curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)  # 接受的参数为数组类型

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
        cv2.rectangle(image_with_roi, top_left, bottom_right, color=(0, 255, 0), thickness=2)

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
def draw_rectangle_roi_base_on_points(image, num_divisions=50, roi_size=20):
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
def analyze_image_with_rois(image, num_divisions=50, roi_size=20, brightness_threshold=50):
    # 调用绘制ROI的函数，获取带有绘制ROI的图像和ROI顶点坐标
    image_with_roi, rois = draw_rectangle_roi_base_on_points(image, num_divisions, roi_size)

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
        # mean_brightness = np.mean(gray[mask == 255], np.float)
        # mean_brightness = round(np.mean(gray[mask == 255], float), 2)
        mean_brightness = round(np.mean(gray[mask == 255], dtype=float), 2)

        # 计算灰度图中ROI区域的最大亮度
        max_brightness = np.max(gray[mask == 255])

        # 计算灰度图中ROI区域的最小亮度
        min_brightness = np.min(gray[mask == 255])

        # 计算亮度低于阈值的像素比例
        # low_brightness_ratio = np.sum(gray[mask == 255] < brightness_threshold) / len(roi_pixels)
        low_brightness_ratio = round(np.sum(gray[mask == 255] < brightness_threshold) / len(roi_pixels), 2)

        # 计算平均颜色
        # 计算ROI区域的平均颜色（BGR）
        mean_color = np.mean(roi_pixels, axis=0).astype(int).tolist()

        # 将结果添加到分析结果列表中
        analysis_results.append({
            'roi_id': idx + 1,  # 添加ROI编号
            'mean_brightness': mean_brightness,
            'max_brightness': max_brightness,
            'min_brightness': min_brightness,
            'mean_color': mean_color,
            'low_brightness_ratio': low_brightness_ratio
        })

    return image_with_roi, analysis_results  # 返回带有绘制ROI的图像和分析结果


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