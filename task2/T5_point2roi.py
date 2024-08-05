import numpy as np
import cv2

def t5point2RotatedRoi(point_and_angel, diameter):
    size = (diameter, diameter)
    # for pt in points_and_angels:
    point, angle = point_and_angel[0], point_and_angel[1]

    # 计算旋转矩形的四个角点
    rect = ((point[0], point[1]), size, angle)
    box = cv2.boxPoints(rect)
    box = np.int32(box)  # 确保为 np.int32 类型

    # 绘制旋转矩形的边界线
    # cv2.polylines(roi_mask, [box], isClosed=True, color=255, thickness=1)

    # 显示结果
    # cv2.imshow("image", image)
    # cv2.imshow("roi_mask", roi_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(f"box:{box}")

    return box

if __name__ == "__main__":
    from T1_contours import t1Contours
    from T2_thin_contours import t2ThinCurve
    from T3_fitted_curve import t3FittedCurve
    from T4_equidistant_point import t4EqualizationPointsAndAngels

    image = cv2.imread(r"E:\workspace\Data\LED_data\task1\6.bmp")
    contours = t1Contours(image)
    fitted_contour = t3FittedCurve(image, contours[0])
    equidistant_points_angels = t4EqualizationPointsAndAngels(image, fitted_contour, segment_length=18)
    box = t5point2RotatedRoi(image, equidistant_points_angels[0], diameter=17)
