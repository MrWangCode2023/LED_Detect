import cv2
import numpy as np
from Object_extraction import object_extraction
from collections import namedtuple

# 入参为二值化图像
def object_curve_fitting(image):
    curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
    filtered_contours, binary, area_count = object_extraction(image)
    binary_image = binary.copy()
    # 细化算法API
    curve_image = cv2.ximgproc.thinning(binary_image)  # 只有一个像素的线
    nonzero_pixels = np.nonzero(curve_image)
    print("nonzero_pixels", len(nonzero_pixels))
    curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
    print("curve_coordinates", curve_coordinates)
    curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)  # 接受的参数为数组类型

    # 使用 namedtuple 打包返回值
    return curve(curve_image, curve_coordinates, curve_length)

if __name__ == "__main__":
    image = cv2.imread('../Data/task1/task1_14.bmp')
    curve = object_curve_fitting(image)
    cv2.imshow('image', image)
    cv2.imshow('skeleton_img', curve.curve_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
