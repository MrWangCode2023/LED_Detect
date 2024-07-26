import cv2
import numpy as np

from F1_RoiPixelExtract import F1RoiPixelExtract

from T7_loadRoiFromJson import t7LoadRoisFromJson
from F2_RoiPixelAnalysis import FroiPixelAnalysis

def FProcessFrame(frame):
    roisPath = r"roisPath.json"
    rois = t7LoadRoisFromJson(roisPath)

    print(f"result[编号， rgb, hsv, 亮暗标签]")
    # for roi in rois:
    for i in range(len(rois)):
        roi = rois[i]

        # 提取单个roi的像素
        roi_image = F1RoiPixelExtract(frame, roi)
        # cv2.imshow("roi_pixel_mask", roi_pixel_mask)

        # 计算每个ROI的RGB均值，HSV
        frameResult = FroiPixelAnalysis(roi_image, i)
        print(f"{frameResult}")

        # 在帧上绘制roi
        box = np.array(roi, dtype=np.int32)
        points_for_polylines = [box]
        cv2.polylines(frame, points_for_polylines, isClosed=True, color=(0, 255, 0), thickness=1)


    # print(f"rois:{len(rois)}")

    # 显示效果
    # cv2.imshow("image", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return frameResult, frame


if __name__ == "__main__":
    image = cv2.imread(r"E:\workspace\Data\LED_data\task1\9.bmp")
    frameResult = FProcessFrame(image)