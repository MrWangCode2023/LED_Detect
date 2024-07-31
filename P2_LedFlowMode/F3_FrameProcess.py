import cv2
import numpy as np

from F1_RoiPixelExtract import F1RoiPixelExtract

from T7_loadRoiFromJson import t7LoadRoisFromJson
from F2_RoiPixelAnalysis import F2roiPixelAnalysis

def F3FrameProcess(frame):
    roisPath = r"roisPath.json"
    rois, _ = t7LoadRoisFromJson(roisPath)
    rois_data = []
    # print(f"roi_nums:{roi_nums}")

    # 遍历单帧中的所有ROI
    for index, roi in enumerate(rois):

        # 提取单个roi的像素
        roi_image = F1RoiPixelExtract(frame, roi)

        # 计算单个ROI的RGB均值，HSV
        roi_data = F2roiPixelAnalysis(roi_image)
        # 插入ROI编号
        roi_data.insert(0, index)

        # 存储单帧所有ROI计算数据
        rois_data.append(roi_data)

        # 在帧上绘制roi
        box = np.array(roi, dtype=np.int32)
        points_for_polylines = [box]
        cv2.polylines(frame, points_for_polylines, isClosed=True, color=(0, 255, 0), thickness=1)

    # 将单帧数据从列表存储成二维数组
    rois_array = np.array(rois_data, dtype=np.float32)
    rois_array = rois_array.reshape(-1, 8)

    return rois_array, frame


if __name__ == "__main__":
    image = cv2.imread(r"E:\workspace\Data\LED_data\task1\9.bmp")
    frameResult = F3FrameProcess(image)