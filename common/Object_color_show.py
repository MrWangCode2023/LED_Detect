import cv2
from common.Common import show_object_color


if __name__ == "__main__":
    color = show_object_color((0, 0, 255))
    cv2.imshow("color show", color)
    cv2.waitKey(0)
    cv2.destroyAllWindows("q")