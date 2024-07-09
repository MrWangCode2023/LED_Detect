import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Mid_point import mid_point



def bbox(contours, image):
    line1_pixel_values = []
    line2_pixel_values = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32)

        a, b, c, d = box[0], box[1], box[2], box[3]
        mid_ab = np.intp(mid_point(a, b))
        mid_cd = np.intp(mid_point(c, d))
        mid_bc = np.intp(mid_point(b, c))
        mid_da = np.intp(mid_point(d, a))

        mask_line1 = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_line2 = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.line(mask_line1, pt1=mid_ab, pt2=mid_cd, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)
        cv2.line(mask_line2, pt1=mid_bc, pt2=mid_da, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)

        line1_pixels = image[mask_line1 == 255]
        line2_pixels = image[mask_line2 == 255]

        line1_pixel_values.append(line1_pixels)
        line2_pixel_values.append(line2_pixels)

    return line1_pixel_values, line2_pixel_values


def mbbox(contours, image):
    line1_pixel_values = []
    line2_pixel_values = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        a, b, c, d = box[0], box[1], box[2], box[3]
        mid_ab = np.intp(mid_point(a, b))
        mid_bc = np.intp(mid_point(b, c))
        mid_cd = np.intp(mid_point(c, d))
        mid_da = np.intp(mid_point(d, a))

        mask_line1 = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_line2 = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.line(mask_line1, pt1=mid_ab, pt2=mid_cd, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)
        cv2.line(mask_line2, pt1=mid_bc, pt2=mid_da, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)

        line1_pixels = image[mask_line1 == 255]
        line2_pixels = image[mask_line2 == 255]

        line1_pixel_values.append(line1_pixels)
        line2_pixel_values.append(line2_pixels)

    return line1_pixel_values, line2_pixel_values