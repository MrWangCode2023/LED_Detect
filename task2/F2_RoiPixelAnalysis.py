import cv2
import numpy as np
from ColorConverter import ColorConverter


def F2roiPixelAnalysis(roi_image, darkThreshold=50):
    """
    计算ROI区域的R、G、B均值

    Args:
        roi_image: 提取的ROI区域图像 (BGR格式)。

    Returns:
        mean_rgb: (R_mean, G_mean, B_mean) 的元组。如果没有有效像素，则返回 (0, 0, 0)。
    """
    tag, r, g, b, h, s, v = 0, 0, 0, 0, 0, 0, 0

    # result = [number]
    result = []

    # 确保roi_image不是空的
    if roi_image is None or roi_image.size == 0:
        result.extend([tag, r, g, b, h, s, v])

        return result

        # 计算非零像素（即 ROI 区域）的数量
    # 返回二维数组（对前两个维度的位置像素在第三维度上进行或操作）
    non_zero_mask = np.any(roi_image != 0, axis=2)
    num_pixels = np.sum(non_zero_mask)

    if num_pixels == 0:
        result.extend([tag, r, g, b, h, s, v])

        return result  # 如果没有有效像素，返回零均值

    # 分别计算R、G、B通道的总和
    sum_r = np.sum(roi_image[..., 2][non_zero_mask])  # BGR中R是第三通道
    sum_g = np.sum(roi_image[..., 1][non_zero_mask])
    sum_b = np.sum(roi_image[..., 0][non_zero_mask])

    # 计算单通道均值
    r = sum_r / num_pixels
    g = sum_g / num_pixels
    b = sum_b / num_pixels

    # 计算RGB值
    rgb = (r, g, b)

    # 计算HSV值
    converter = ColorConverter()
    h, s, v = converter.rgb2hsv(rgb)

    # 判断当前roi是否为暗
    tag = 1 if h > darkThreshold else 0

    # tag, r, g, b, h, s, v = \
    #     round(tag, 0), round(r, 4), round(g, 4), round(b, 4), round(h, 4), round(s, 4), round(v, 4)

    result.extend([tag, r, g, b, h, s, v])

    return result


if __name__ == "__main__":
    from F1_RoiPixelExtract import F1RoiPixelExtract

    # 示例用法
    # image = cv2.imread(r"E:\workspace\Data\LED_data\task4\27.png")
    image = cv2.imread(r"E:\workspace\Data\LED_data\task2\4.bmp")
    roi = [(50, 50), (150, 50), (150, 150), (50, 150)]  # 设定四边形顶点
    roi_image = F1RoiPixelExtract(image, roi)

    F2roiPixelAnalysis(roi_image, 0)

    # cv2.imshow("roi_image", roi_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
