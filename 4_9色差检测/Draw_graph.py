import cv2
import numpy as np
from Show import show_image
from Object_extraction import object_extraction

def draw_graph(image, contours):
    BBox_img = image.copy()
    MBox_img = image.copy()
    object_positions = []

    for idx, contour in enumerate(contours):
        # 计算BBox
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(BBox_img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 绿色矩形框，线条宽度为2
        # 在BBox的中心绘制编号
        bbox_center = (x + w // 2, y + h // 2)
        cv2.putText(BBox_img, str(idx), bbox_center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

        # 计算MBox
        rect = cv2.minAreaRect(contour)
        center = rect[0]  # (x, y)中心坐标
        box = cv2.boxPoints(rect)  # 获取矩形的四个顶点
        box = np.intp(box)

        # 绘制最小外接矩形
        cv2.drawContours(MBox_img, [box], 0, (0, 255, 0), 1)  # 绿色矩形框，线条宽度为2
        # 在MBox的中心绘制编号
        mbox_center = (int(center[0]-10), int(center[1]))
        cv2.putText(MBox_img, str(idx), mbox_center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

        # 中心点
        object_positions.append(mbox_center)  # 使用最小外接矩形的中心坐标

    # 确保对象位置为numpy数组
    object_positions = np.array(object_positions)

    # image_dict = {
    #     "Original": image,
    #     "BBoxes": BBox_img,
    #     "MBoxes": MBox_img
    # }
    # show_image(image_dict)

    result = (
        # BBox_img,
        MBox_img,
        object_positions
    )

    return result

if __name__ == "__main__":
    image = cv2.imread("E:\workspace\Data\LED_data\\4_9\\1.png")
    filtered_contours = object_extraction(image)
    result = draw_graph(image, filtered_contours)
