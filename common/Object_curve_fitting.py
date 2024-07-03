import cv2
from common.Common import object_curve_fitting


if __name__ == "__main__":
    image = cv2.imread('../../Data/LED_data/task2/02.bmp')
    # E:\workspace\Data\LED_data\task2
    curve = object_curve_fitting(image)
