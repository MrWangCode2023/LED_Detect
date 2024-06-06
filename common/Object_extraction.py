import cv2
import numpy as np
from common.Common import object_extraction


if __name__ == "__main__":
    # 示例图像路径
    image_path = "../Data/task1/task1_12.bmp"  # 替换为实际图像路径
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or unable to read")
        exit()

    contours, binary, ROI_count = object_extraction(image)

    # 显示结果
    cv2.imshow("binary", binary)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

######################################################################


