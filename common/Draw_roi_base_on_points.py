import cv2
import numpy as np
from Common import curve_division, object_curve_fitting


###################################################################################################
# 绘制圆形ROI
def draw_circle_roi_base_on_points(image, num_divisions=50, roi_size=20):
    """
    Draw ROI (Region of Interest) circles on an image centered at each point.

    Parameters:
    - image: The input image
    - num_divisions: Number of divisions for the curve
    - roi_size: Diameter of the ROI circles (default is 20)

    Returns:
    - image_with_roi: Image with the ROI circles drawn
    """
    curve = object_curve_fitting(image)  # 获取曲线数据
    points_and_angles = curve_division(curve.curve_length, curve.fitted_contour, num_divisions)
    image_with_roi = image.copy()
    radius = roi_size // 2  # 半径是直径的一半

    for (point, angle) in points_and_angles:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image_with_roi, center, radius, color=(0, 255, 0), thickness=2)

    return image_with_roi  # 返回绘制了ROI的图像


# 绘制矩形ROI
def draw_rectangle_roi_base_on_points(image, num_divisions=50, roi_size=20):
    """
    Draw ROI (Region of Interest) rectangles on an image centered at each point with the specified angle.

    Parameters:
    - image: The input image
    - num_divisions: Number of divisions for the curve
    - roi_size: Size of the ROI rectangles (default is 20)

    Returns:
    - image_with_roi: Image with the ROI rectangles drawn
    - rois: List of extracted ROI regions
    """
    curve = object_curve_fitting(image)  # 获取曲线数据
    points_and_angles = curve_division(curve.curve_length, curve.fitted_contour, num_divisions)
    image_with_roi = image.copy()
    rois = []

    for (point, angle) in points_and_angles:
        center = (int(point[0]), int(point[1]))
        size = (roi_size, roi_size)

        # 绘制旋转矩形
        rect = (center, size, angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(image_with_roi, [box], 0, (255, 255, 255), 2)

        # 提取旋转后的ROI区域像素
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [box], 0, (255), thickness=cv2.FILLED)
        roi = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(box)  # 提取ROI外接矩形区域

        # 检查ROI的边界
        if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
            roi_cropped = roi[y:y + h, x:x + w]
            rois.append(roi_cropped)
        else:
            print(f"ROI at ({center}) with size ({w}, {h}) is out of image bounds and will be skipped.")

    return image_with_roi, rois




################################################################################################
if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')

    # 旋转后的ROI矩形
    image_with_rois, rois = draw_rectangle_roi_base_on_points(image, num_divisions=30, roi_size=20)
    # ROI圆形
    # image_with_rois = draw_circle_roi_base_on_points(image, 50, roi_size=20)

    # 显示结果图像
    cv2.imshow('Divided Curve with ROI', image_with_rois)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 打印或处理提取的ROI区域像素
    for i, roi in enumerate(rois):
        if roi.size > 0:
            cv2.imshow(f"ROI {i + 1}", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(f"ROI {i + 1} 像素值：", roi)
        else:
            print(f"ROI {i + 1} 是空的，未显示")

