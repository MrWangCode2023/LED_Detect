import cv2
import numpy as np
import matplotlib.pyplot as plt
from Object_extraction import object_extraction
from Show import show_image
from Point_cloud_segmentation import point_cloud_segmentation
from DTVn_detect import dtv_detect

def main(image):
    image_dict1 = point_cloud_segmentation(image, k=4)
    image_dict2, DTV_dict = dtv_detect(image_dict1["Points_segment_image"])
    image_dict = {**image_dict1, **image_dict2}
    show_image(image_dict)
    print(f"畸变值检测结果： {DTV_dict}")


if __name__ == "__main__":
    # 读取图像
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\25.png")
    # 调用函数并显示结果
    main(image)

