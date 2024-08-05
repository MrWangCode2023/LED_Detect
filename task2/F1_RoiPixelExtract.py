import cv2
import numpy as np


def F1RoiPixelExtract(image, roi):
    """
    根据四边形顶点提取 ROI 区域像素。

    参数：
    - image: 输入图像
    - roi: 四边形顶点的列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    返回：
    - roi_image: 提取的 ROI 区域图像
    """
    # 将四边形顶点转换为多边形轮廓（列表中的单个元素是一个点的数组）
    pts = np.array(roi, np.int32)
    pts = pts.reshape((-1, 1, 2))  # 重塑为 (n_points, 1, 2) 的形状

    # 创建一个与图像同大小的掩码，并填充多边形区域
    roiMask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roiMask, [pts], (255, 255, 255))

    # 使用掩码提取 ROI
    roi_image = cv2.bitwise_and(image, image, mask=roiMask)

    # 显示结果
    # cv2.imshow("roi_image", roi_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return roi_image


if __name__ == "__main__":
    image = cv2.imread(r"E:\workspace\Data\LED_data\task1\6.bmp")
    roi = [[2, 533], [15, 522], [26, 534], [13, 546]]
    roi_image = F1RoiPixelExtract(image, roi)