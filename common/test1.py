import cv2
import numpy as np


def draw_rectangle_roi_base_on_points(image, points_and_angles, roi_size=20):
    """
    在图像上绘制以每个点为中心，并根据指定角度旋转的矩形ROI区域。

    Parameters:
    - image: 输入图像
    - points_and_angles: 包含坐标点和角度的列表，格式 [(point1, angle1), (point2, angle2), ..., (pointn, anglen)]
    - roi_size: ROI矩形区域的尺寸（默认是20）

    Returns:
    - image_with_roi: 绘制了ROI矩形区域的图像
    - rois: 提取的ROI区域像素列表
    """
    image_with_roi = image.copy()
    half_size = roi_size // 2
    rois = []

    for (point, angle) in points_and_angles:
        center = (int(point[0]), int(point[1]))
        size = (roi_size, roi_size)

        # 旋转矩形
        rect = (center, size, angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(image_with_roi, [box], 0, (255, 255, 255), 2)

        # 提取旋转后的ROI区域像素
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [box], 0, (255), thickness=cv2.FILLED)
        roi = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(box)
        roi_cropped = roi[y:y + h, x:x + w]
        rois.append(roi_cropped)

    return image_with_roi, rois


# 示例代码
if __name__ == "__main__":
    # 读取输入图像
    image = cv2.imread('../Data/task1/task1_13.bmp')

    # 定义矩形的中心点、尺寸和角度
    points_and_angles = [
        (np.array([425.24548175, 846.75451825]), -143.50432798364128),
        (np.array([406.98265277, 872.01734723]), -144.33533930347318),
        (np.array([394.74683828, 889.25316172]), -143.5300010143727),
        (np.array([378.64628695, 910.35371305]), -142.92610596251635),
        (np.array([363.83398124, 930.16601876]), -143.9552079161335),
        (np.array([346.57115225, 954.42884775]), -144.9987519480493),
        (np.array([333.46595332, 973.53404668]), -144.3462026077095),
        (np.array([318.6536476, 993.3463524]), -142.02947300181845),
        (np.array([301.16979547, 1014.91510226]), -155.2768632851831),
        (np.array([304.99998317, 1023.0]), 154.6509341809821)
    ]

    # 绘制并提取旋转后的ROI区域
    image_with_roi, rois = draw_rectangle_roi_base_on_points(image, points_and_angles)

    # 显示绘制了ROI矩形区域的图像
    cv2.imshow("Image with ROIs", image_with_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 打印或处理提取的ROI区域像素
    for i, roi in enumerate(rois):
        cv2.imshow(f"ROI {i + 1}", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"ROI {i + 1} 像素值：", roi)
