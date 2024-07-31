import cv2
import numpy as np

from T1_contours import t1Contours
from T2_thin_contours import t2ThinCurve
from T3_fitted_curve import t3FittedCurve
from T4_equidistant_point import t4EqualizationPointsAndAngels
from T5_point2roi import t5point2RotatedRoi
from T6_saveRoi2Json import t6saveBoxes2Json


def templateRois(image):
    # roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    rois = []
    roisPath = r"roisPath.json"

    contours = t1Contours(image)
    # curve = t2ThinCurve(image, contours[0])
    curve = t3FittedCurve(image, contours[0])
    equidistant_points_angels = t4EqualizationPointsAndAngels(image, curve, 30)
    for pt_agl in equidistant_points_angels:
        roi = t5point2RotatedRoi(pt_agl, diameter=25)
        rois.append(roi)

        # 绘制roi
        # cv2.drawContours(roi_mask, [roi], 0, (255, 255, 255), 1)

    # 存储ROIs
    t6saveBoxes2Json(rois, roisPath)

    # 打印结果
    # print(f"roi个数：{len(rois)}")
    # print(f"roi坐标：{rois}")

    # 显示效果
    # cv2.imshow("roi_mask", roi_mask)
    # cv2.imshow("roi_mask", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return rois

if __name__ == "__main__":
    # template_path = r"E:\workspace\Data\LED_data\task2\3.bmp
    template_path = r"E:\workspace\Data\LED_data\task2\7.png"
    image = cv2.imread(template_path)
    template_roi = templateRois(image)
