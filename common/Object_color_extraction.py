import cv2
from common.Common import object_color_extraction, show_object_color


if __name__ == '__main__':
    img = cv2.imread('../../Data/LED_data/task1/task1_12.bmp')
    data = object_color_extraction(img)
    object_color = show_object_color(data.ROI_BGR_mean)
    cv2.imshow("object_color", object_color)
    print(data.ROI_count)
    cv2.waitKey(0)
    cv2.destroyAllWindows()