import numpy as np
import cv2

def roi_generation(image, relative_center_position):
    # 获取图像的尺寸
    H, V, _ = image.shape
    scale = min(H, V)

    # 创建一个图像副本用于绘制ROI
    image_with_rois = image.copy()
    roi_mask = np.zeros_like(image)

    for center_coordinate in relative_center_position:
        x1, y1 = center_coordinate
        x, y = int(x1 * H), int(y1 * V)
        radius = int(scale * 0.02)

        # 在图像副本上绘制圆形ROI
        cv2.circle(image_with_rois, (x, y), radius, (0, 0, 255), 1)
        # 实心圆
        cv2.circle(roi_mask, (x, y), radius, (255, 255, 255), -1)

    cv2.imshow('ROI Image', image_with_rois)
    cv2.imshow('roi_mask', roi_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image_with_rois, roi_mask


if __name__ == "__main__":
    # 创建一个测试图像（白色背景）
    image = cv2.imread("E:\workspace\Data\LED_data\\task4\\4.bmp")
    # 定义中心坐标
    center_coordinates = [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4), (0.7, 0.7)]
    # 运行测试
    # 生成并显示包含ROI的图像
    result_image = roi_generation(image, center_coordinates)