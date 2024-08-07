import cv2
import numpy as np
from RGB_to_CIE import rgb_to_cie_xy
from common.Common import show_image
from CIE_to_temperature import cie_xy_to_CCT
from common.Plot_cie_chromaticity_diagram import plot_cie_chromaticity_diagram
from generate_relative_points import generate_relative_points


def color_temperature_detect(image):
    # 获取图像的尺寸
    H, V, _ = image.shape
    scale = min(H, V)

    image_with_rois = image.copy()
    RGBs, CIEs, CCTs = [], [], []

    # 根据图像尺寸生成相对位置点位
    relative_positions = generate_relative_points()
    for relative_position in relative_positions:
        x1, y1 = relative_position
        x, y = int(x1 * H), int(y1 * V)
        radius = int(scale * 0.02)

        # 生成ROI
        cv2.circle(image_with_rois, (x, y), radius, (255, 0, 0), 1)

        # 提取ROI像素
        roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(roi_mask, (x, y), radius, (255, 255, 255), -1)
        roi_image = cv2.bitwise_and(image, image, mask=roi_mask)

        # 计算ROI区域RGB均值
        nonzero_pixel_count = float(np.count_nonzero(roi_mask))
        if nonzero_pixel_count == 0:
            continue
        blue_channel = roi_image[:, :, 0]
        green_channel = roi_image[:, :, 1]
        red_channel = roi_image[:, :, 2]
        blue_sum = np.sum(blue_channel)
        green_sum = np.sum(green_channel)
        red_sum = np.sum(red_channel)
        # 分别计算每个通道的均值
        if nonzero_pixel_count > 0:
            blue_mean = blue_sum / nonzero_pixel_count
            green_mean = green_sum / nonzero_pixel_count
            red_mean = red_sum / nonzero_pixel_count
        else:
            blue_mean = green_mean = red_mean = 0
        # RGB
        rgb = (red_mean, green_mean, blue_mean)
        # print("rgb:", rgb)
        RGBs.append(rgb)

        # CIE
        cie = rgb_to_cie_xy(rgb)
        # print("cie:", cie)
        CIEs.append(cie)

        # CIE to color_temperature
        cct = cie_xy_to_CCT(cie)
        CCTs.append(cct)

    fig, ax = plot_cie_chromaticity_diagram(CIEs)
    for cct in CCTs:
        print("ROIs区域色温值（开尔文）：", cct, "K")

    show_dict = {
        'ROI Image': image_with_rois,
    }
    show_image(show_dict)

    return image_with_rois

if __name__ == "__main__":
    # 创建一个测试图像（白色背景）
    image = cv2.imread("E:\workspace\Data\LED_data\\task4\\4.bmp")


    result_image = color_temperature_detect(image)
