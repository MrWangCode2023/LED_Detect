import cv2
from common.Common import object_curve_fitting


if __name__ == "__main__":
    image = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')
    curve = object_curve_fitting(image)
