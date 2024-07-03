import cv2
from common.Common import object_color_extraction, show_object_color


if __name__ == '__main__':
    img = cv2.imread('../../Data/LED_data/task1/task1_12.bmp')
    data = object_color_extraction(img)
    print(data.ROI_count)
