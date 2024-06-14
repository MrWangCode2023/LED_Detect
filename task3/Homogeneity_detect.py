import cv2
import numpy as np
from common.Common import analyze_image_with_rois, draw_rectangle_roi_base_on_divisions


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



def homogeneity_detect(image_path):
    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    if image is None:
        raise ValueError(f"图像加载失败: {image_path}")

    # 绘制ROI并分析
    image_with_roi, analysis_results = analyze_image_with_rois(image, num_divisions=45, roi_size=20, brightness_threshold=50)

    # 打印结果
    for result in analysis_results:
        print(result)
    # 显示结果图像
    cv2.imshow('Image with ROIS', image_with_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows("q")


if __name__ == '__main__':
    image_path = '../../Data/LED_data/task1/task1_13.bmp'
    homogeneity_detect(image_path)
