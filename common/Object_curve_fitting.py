from collections import namedtuple

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from common.Common import object_extraction, show_image


def object_curve_fitting(image):
    # cv2.imshow('image', image)
    curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
    filtered_contours, binary, roi_count = object_extraction(image)
    binary_image = binary.copy()

    # 细化算法API
    skeleton_image = cv2.ximgproc.thinning(binary_image)  # 只有一个像素的线
    skeleton = skeletonize(skeleton_image // 255)  # Convert to boolean and skeletonize
    curve_image = img_as_ubyte(skeleton)

    nonzero_pixels = np.nonzero(curve_image)

    # 如果没有检测到曲线，返回None
    if len(nonzero_pixels[0]) == 0:
        return curve(None, None, None)

    curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
    curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)  # 接受的参数为数组类型

    image_dict = {
        # "Image": image,
        # 'binary': binary_image,
        'curve_img': curve_image,
    }
    show_image(image_dict)

    return curve(curve_image, curve_coordinates, curve_length)

if __name__ == "__main__":
    image = cv2.imread(r'../../Data/LED_data/task1/task1_13.bmp')
    # E:\workspace\Data\LED_data\task2
    curve = object_curve_fitting(image)
