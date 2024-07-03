import cv2
import numpy as np
from common.Common import draw_rectangle_roi_base_on_divisions, rgb_to_cie_xy
from common.Common import display_images_with_titles


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
    display_images_with_titles(image_dict)

    return image_with_roi, analysis_results  # 返回带有绘制ROI的图像和分析结果


if __name__ == '__main__':
    image = cv2.imread(r'../../Data/LED_data/task2/02.bmp')
    homogeneity_detect(image)
