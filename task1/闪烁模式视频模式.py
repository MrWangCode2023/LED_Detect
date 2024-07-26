from collections import namedtuple

import cv2
import numpy as np
import time

from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from common.Common import curve_division


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

    edges = cv2.Canny(blurred, threshold1=80, threshold2=240)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask1 = np.zeros(img.shape[:2], dtype=np.uint8)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= 10:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 闭合操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    iterations = 10
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # 在此处绘制所有过滤后的轮廓
    binary = closed.copy()
    contours1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask1, contours1, -1, 255, thickness=cv2.FILLED)
    cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color,
                  border_thickness)

    roi_count = len(filtered_contours)
    print("Number of Objects:", roi_count)

    # image_dict = {
    #     "Image": image,
    #     "Edges": edges,
    #     "Mask": mask,
    #     "Mask1": mask1,
    #     "Binary Image": closed
    # }
    # display_images_with_titles(image_dict)

    return filtered_contours, binary, roi_count

def object_curve_fitting(image):
    # cv2.imshow('image', image)
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

    image_dict = {
        "Image": image,
        'binary': binary_image,
        'curve_img': curve_image,
    }
    # show_image(image_dict)

    return curve(curve_image, curve_coordinates, curve_length)

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

def rgb_to_cie_xy(rgb):
    # sRGB to Linear RGB(伽马矫正)
    def linearize(c):
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4

    # Apply gamma correction to each channel
    linear_rgb = [linearize(c) for c in rgb]

    # Conversion matrix from RGB to XYZ
    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    # Convert linear RGB to XYZ
    xyz = np.dot(mat, linear_rgb)

    # Extract X, Y, Z components
    X, Y, Z = xyz

    # Calculate x and y chromaticity coordinates
    denom = X + Y + Z
    if denom == 0:
        return 0, 0
    x = X / denom
    y = Y / denom

    return x, y

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
        mean_color_xyz = rgb_to_cie_xy(mean_color_array)

        # 将结果添加到分析结果列表中
        analysis_result = {
            'roi_id': idx + 1,  # 添加ROI编号
            'mean_brightness': mean_brightness,
            'max_brightness': max_brightness,
            'min_brightness': min_brightness,
            'ROI_color_RGB': mean_color_rgb,
            'low_brightness_ratio': low_brightness_ratio,
            'ROI_CIE1931_xyz': np.round(mean_color_xyz, 4).tolist()  # 添加CIE 1931 XYZ值
        }
        analysis_results.append(analysis_result)

        # 打印每个ROI的分析结果
        print(f"ROI {idx + 1} analysis result:", analysis_result)

    # image_dict = {
    #     "Imag": image,
    #     "Image_with_roi": image_with_roi
    # }
    # display_images_with_titles(image_dict)

    return image_with_roi, analysis_results  # 返回带有绘制ROI的图像和分析结果

def detect_LED_blinking_from_video(video_path, roi_size):
    """
    从视频文件中读取帧并检测LED闪烁。

    参数：
    video_path -- 视频文件路径
    roi_size -- ROI的大小
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    roi_status = {}
    start_time = time.time()
    end_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, analysis_results = analyze_image_with_rois(frame, roi_size=roi_size)

        current_time = time.time()
        on_rois = []
        unlit_rois = []
        for result in analysis_results:
            roi_id = result['roi_id']
            mean_brightness = result['mean_brightness']

            if mean_brightness > 50:
                on_rois.append(roi_id)
                if roi_id not in roi_status:
                    roi_status[roi_id] = 'on'
                    if all([roi_status.get(i, 'off') == 'on' for i in range(1, roi_id)]):
                        end_time = time.time()
                    else:
                        print(f"Error in lighting sequence at ROI {roi_id}")
            else:
                unlit_rois.append(roi_id)
                if roi_id in roi_status and roi_status[roi_id] == 'on':
                    print(f"Error: ROI {roi_id} turned off")

        all_on = all(status == 'on' for status in roi_status.values())
        if all_on:
            end_time = time.time()

        if end_time is not None:
            total_time = end_time - start_time
        else:
            total_time = current_time - start_time

        unlit_rois = [i for i in range(1, max(roi_status.keys()) + 1) if roi_status.get(i, 'off') == 'off']

        # 定义每行文本的高度
        line_height = 30

        # 计算每个文本块的行数
        lighted_text_lines = f"Lighted ROI indexes:\n{on_rois}".split('\n')
        unlighted_text_lines = f"Unlighted ROI indexes:\n{unlit_rois}".split('\n')
        total_time_text_lines = f"Total time:\n{total_time:.2f} seconds".split('\n')

        # 绘制点亮的ROI索引
        y_start = 30
        for idx, line in enumerate(lighted_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制未点亮的ROI索引，调整起始位置以避免重叠
        y_start += len(lighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
        for idx, line in enumerate(unlighted_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 绘制总时间，调整起始位置以避免重叠
        y_start += len(unlighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
        for idx, line in enumerate(total_time_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2)

        # 在终端上显示结果
        print(f"Lighted ROIs: {on_rois}")
        print(f"Unlit ROIs: {unlit_rois}")
        print(f"Total time: {total_time:.2f} seconds")

        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = "E:\\workspace\\Data\\LED_data\\task1\\2.avi"  # 替换为实际的视频文件路径
    detect_LED_blinking_from_video(video_path, roi_size=20)
