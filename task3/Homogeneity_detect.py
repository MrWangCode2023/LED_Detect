from collections import namedtuple

import cv2
import numpy as np
from skimage.util import img_as_ubyte

from common.Common import draw_rectangle_roi_base_on_divisions, rgb_to_cie_xy, object_extraction, skeletonize, curve_division
from Show import show_image


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
    show_image(image_dict)

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

def homogeneity_detect(image, num_divisions=20, roi_size=20, brightness_threshold=50):
    if image is None:
        raise ValueError(f"图像加载失败: image is None")

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
            'ROI_CIE1931_xy': np.round(mean_color_xyz, 4).tolist()  # 添加CIE 1931 XYZ值
        }
        analysis_results.append(analysis_result)

        # 打印每个ROI的分析结果
        print(f"ROI {idx + 1} analysis result:", analysis_result)

    image_dict = {
        "Imag": image,
        "Image_with_roi": image_with_roi
    }
    show_image(image_dict)

    return image_with_roi, analysis_results  # 返回带有绘制ROI的图像和分析结果


if __name__ == '__main__':
    image = cv2.imread(r'../../Data/LED_data/task1/task1_13.bmp')
    homogeneity_detect(image, num_divisions=40, roi_size=20, brightness_threshold=50)
