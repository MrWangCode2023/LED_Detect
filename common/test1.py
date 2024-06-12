# import cv2
# from collections import namedtuple
# import numpy as np
#
# def object_extraction(image):
#     img = np.copy(image)
#     border_color = (0, 0, 0)
#     border_thickness = 7
#     height, width, _ = img.shape
#     border_top = border_left = 0
#     border_bottom = height - 1
#     border_right = width - 1
#     cv2.rectangle(img, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)
#     blurred = cv2.GaussianBlur(img, (3, 3), 0)
#     edges = cv2.Canny(blurred, threshold1=120, threshold2=240)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     mask = np.zeros(img.shape[:2], dtype=np.uint8)
#     filtered_contours = []
#     for contour in contours:
#         if cv2.contourArea(contour) >= 800:
#             filtered_contours.append(contour)
#             cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     iterations = 3
#     closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
#     binary = closed.copy()
#     cv2.drawContours(binary, filtered_contours, -1, 255, thickness=cv2.FILLED)
#     cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)
#     roi_count = len(filtered_contours)
#     return filtered_contours, binary, roi_count
#
# def object_curve_fitting(image):
#     curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
#     filtered_contours, binary, roi_count = object_extraction(image)
#     binary_image = binary.copy()
#
#     # 细化算法API
#     curve_image = cv2.ximgproc.thinning(binary_image)  # 只有一个像素的线
#     nonzero_pixels = np.nonzero(curve_image)
#
#     # 如果没有检测到曲线，返回None
#     if len(nonzero_pixels[0]) == 0:
#         return curve(None, None, None)
#
#     curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
#     curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)  # 接受的参数为数组类型
#
#     # cv2.imshow('image', image)
#     # cv2.imshow('binary', binary_image)
#     # cv2.imshow('curve_img', curve_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     return curve(curve_image, curve_coordinates, curve_length)
#
# def is_continuous_and_uniform(curve_image):
#     if curve_image is None:
#         return False, False
#     nonzero_pixels = np.column_stack(np.nonzero(curve_image))
#     height, width = curve_image.shape
#     is_continuous = True
#     for y, x in nonzero_pixels:
#         neighborhood = curve_image[max(0, y - 1):min(y + 2, height), max(0, x - 1):min(x + 2, width)]
#         if np.sum(neighborhood) < 2:
#             is_continuous = False
#             break
#     dist_transform = cv2.distanceTransform(cv2.bitwise_not(curve_image), cv2.DIST_L2, 5)
#     max_dist = np.max(dist_transform)
#     is_uniform = max_dist <= 1.0
#     return is_continuous, is_uniform
#
# if __name__ == "__main__":
#     image = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')
#     curve = object_curve_fitting(image)
#     is_continuous, is_uniform = is_continuous_and_uniform(curve.curve_image)
#     print(f"Curve Image: {curve.curve_image is not None}")
#     print(f"Curve Coordinates: {curve.curve_coordinates is not None}")
#     print(f"Curve Length: {curve.curve_length}")
#     print(f"Is Continuous: {is_continuous}")
#     print(f"Is Uniform: {is_uniform}")

##################
# import cv2
# from collections import namedtuple
# import numpy as np
#
#
# def object_extraction(image):
#     img = np.copy(image)
#     border_color = (0, 0, 0)
#     border_thickness = 7
#     height, width, _ = img.shape
#     border_top = border_left = 0
#     border_bottom = height - 1
#     border_right = width - 1
#     cv2.rectangle(img, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)
#
#     # 图像预处理
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#     edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
#
#     # 使用自适应阈值
#     binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     mask = np.zeros(img.shape[:2], dtype=np.uint8)
#     filtered_contours = []
#     for contour in contours:
#         if cv2.contourArea(contour) >= 800:
#             filtered_contours.append(contour)
#             cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#
#     # 形态学操作
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     iterations = 3
#     closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
#     binary = closed.copy()
#     cv2.drawContours(binary, filtered_contours, -1, 255, thickness=cv2.FILLED)
#     cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)
#     roi_count = len(filtered_contours)
#
#     return filtered_contours, binary, roi_count
#
#
# def object_curve_fitting(image):
#     curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
#     filtered_contours, binary, roi_count = object_extraction(image)
#     binary_image = binary.copy()
#
#     # 细化算法API
#     binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # 确保是三通道图像
#     curve_image = cv2.ximgproc.thinning(cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY))  # 细化为单像素曲线
#     nonzero_pixels = np.nonzero(curve_image)
#
#     # 如果没有检测到曲线，返回None
#     if len(nonzero_pixels[0]) == 0:
#         return curve(None, None, None)
#
#     curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
#     curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)  # 接受的参数为数组类型
#
#     return curve(curve_image, curve_coordinates, curve_length)
#
#
# def is_continuous_and_uniform(curve_image):
#     if curve_image is None:
#         return False, False
#     nonzero_pixels = np.column_stack(np.nonzero(curve_image))
#     height, width = curve_image.shape
#     is_continuous = True
#     for y, x in nonzero_pixels:
#         neighborhood = curve_image[max(0, y - 1):min(y + 2, height), max(0, x - 1):min(x + 2, width)]
#         if np.sum(neighborhood) < 2:
#             is_continuous = False
#             break
#     dist_transform = cv2.distanceTransform(cv2.bitwise_not(curve_image), cv2.DIST_L2, 5)
#     max_dist = np.max(dist_transform)
#     is_uniform = max_dist <= 1.0
#     return is_continuous, is_uniform
#
#
# if __name__ == "__main__":
#     image = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')
#     curve = object_curve_fitting(image)
#     is_continuous, is_uniform = is_continuous_and_uniform(curve.curve_image)
#     print(f"Curve Image: {curve.curve_image is not None}")
#     print(f"Curve Coordinates: {curve.curve_coordinates is not None}")
#     print(f"Curve Length: {curve.curve_length}")
#     print(f"Is Continuous: {is_continuous}")
#     print(f"Is Uniform: {is_uniform}")
########
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# from skimage import img_as_ubyte
# from collections import namedtuple
# from scipy.interpolate import splprep, splev
# import matplotlib.pyplot as plt
#
#
# def object_extraction(image):
#     img = np.copy(image)
#     border_color = (0, 0, 0)
#     border_thickness = 7
#     height, width, _ = img.shape
#     border_top = border_left = 0
#     border_bottom = height - 1
#     border_right = width - 1
#     cv2.rectangle(img, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#     edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
#
#     binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     mask = np.zeros(img.shape[:2], dtype=np.uint8)
#     filtered_contours = []
#     for contour in contours:
#         if cv2.contourArea(contour) >= 800:
#             filtered_contours.append(contour)
#             cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     iterations = 3
#     closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
#     binary = closed.copy()
#     cv2.drawContours(binary, filtered_contours, -1, 255, thickness=cv2.FILLED)
#     cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)
#     roi_count = len(filtered_contours)
#
#     return filtered_contours, binary, roi_count
#
#
# def object_curve_fitting(image):
#     curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
#     filtered_contours, binary, roi_count = object_extraction(image)
#     binary_image = binary.copy()
#
#     skeleton = skeletonize(binary_image // 255)  # Skeletonize the binary image
#     curve_image = img_as_ubyte(skeleton)
#     nonzero_pixels = np.nonzero(curve_image)
#
#     if len(nonzero_pixels[0]) == 0:
#         return curve(None, None, None)
#
#     curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
#     curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)
#
#     return curve(curve_image, curve_coordinates, curve_length)
#
#
# def is_continuous_and_uniform(curve_image):
#     if curve_image is None:
#         return False, False
#     nonzero_pixels = np.column_stack(np.nonzero(curve_image))
#     height, width = curve_image.shape
#     is_continuous = True
#     for y, x in nonzero_pixels:
#         neighborhood = curve_image[max(0, y - 1):min(y + 2, height), max(0, x - 1):min(x + 2, width)]
#         if np.sum(neighborhood) < 2:
#             is_continuous = False
#             break
#     dist_transform = cv2.distanceTransform(cv2.bitwise_not(curve_image), cv2.DIST_L2, 5)
#     max_dist = np.max(dist_transform)
#     is_uniform = max_dist <= 1.0
#     return is_continuous, is_uniform
#
#
# def fit_b_spline(curve_coordinates):
#     if curve_coordinates is None:
#         return None
#     tck, u = splprep([curve_coordinates[:, 0], curve_coordinates[:, 1]], s=3)
#     u_new = np.linspace(u.min(), u.max(), 1000)
#     x_new, y_new = splev(u_new, tck)
#     return np.column_stack((x_new, y_new))
#
#
# if __name__ == "__main__":
#     image = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')
#     curve = object_curve_fitting(image)
#
#     if curve.curve_coordinates is not None:
#         fitted_curve = fit_b_spline(curve.curve_coordinates)
#         is_continuous, is_uniform = is_continuous_and_uniform(curve.curve_image)
#
#         print(f"Curve Image: {curve.curve_image is not None}")
#         print(f"Curve Coordinates: {curve.curve_coordinates is not None}")
#         print(f"Curve Length: {curve.curve_length}")
#         print(f"Is Continuous: {is_continuous}")
#         print(f"Is Uniform: {is_uniform}")
#
#         plt.figure()
#         plt.imshow(curve.curve_image, cmap='gray')
#         plt.plot(fitted_curve[:, 0], fitted_curve[:, 1], 'r-', linewidth=2)
#         plt.title("Fitted B-Spline Curve")
#         plt.show()
#
#     else:
#         print("No curve detected")
#########
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from collections import namedtuple
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


def object_extraction(image):
    img = np.copy(image)
    border_color = (0, 0, 0)
    border_thickness = 7
    height, width, _ = img.shape
    border_top = border_left = 0
    border_bottom = height - 1
    border_right = width - 1
    cv2.rectangle(img, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= 800:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iterations = 3
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    binary = closed.copy()
    cv2.drawContours(binary, filtered_contours, -1, 255, thickness=cv2.FILLED)
    cv2.rectangle(binary, (border_left, border_top), (border_right, border_bottom), border_color, border_thickness)
    roi_count = len(filtered_contours)

    return filtered_contours, binary, roi_count


def object_curve_fitting(image):
    curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
    filtered_contours, binary, roi_count = object_extraction(image)
    binary_image = binary.copy()

    skeleton = skeletonize(binary_image // 255)  # Skeletonize the binary image
    curve_image = img_as_ubyte(skeleton)
    nonzero_pixels = np.nonzero(curve_image)

    if len(nonzero_pixels[0]) == 0:
        return curve(None, None, None)

    curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
    curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)

    return curve(curve_image, curve_coordinates, curve_length)


def is_continuous_and_uniform(curve_image):
    if curve_image is None:
        return False, False
    nonzero_pixels = np.column_stack(np.nonzero(curve_image))
    height, width = curve_image.shape
    is_continuous = True
    for y, x in nonzero_pixels:
        neighborhood = curve_image[max(0, y - 1):min(y + 2, height), max(0, x - 1):min(x + 2, width)]
        if np.sum(neighborhood) < 2:
            is_continuous = False
            break
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(curve_image), cv2.DIST_L2, 5)
    max_dist = np.max(dist_transform)
    is_uniform = max_dist <= 1.0
    return is_continuous, is_uniform


def fit_b_spline(curve_coordinates):
    if curve_coordinates is None:
        return None
    tck, u = splprep([curve_coordinates[:, 0], curve_coordinates[:, 1]], s=3)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new))


if __name__ == "__main__":
    image = cv2.imread('../../Data/LED_data/task1/task1_13.bmp')
    curve = object_curve_fitting(image)

    if curve.curve_coordinates is not None:
        fitted_curve = fit_b_spline(curve.curve_coordinates)
        is_continuous, is_uniform = is_continuous_and_uniform(curve.curve_image)

        print(f"Curve Image: {curve.curve_image is not None}")
        print(f"Curve Coordinates: {curve.curve_coordinates is not None}")
        print(f"Curve Length: {curve.curve_length}")
        print(f"Is Continuous: {is_continuous}")
        print(f"Is Uniform: {is_uniform}")

        plt.figure()
        plt.imshow(curve.curve_image, cmap='gray')
        plt.plot(fitted_curve[:, 0], fitted_curve[:, 1], 'r-', linewidth=2)
        plt.title("Fitted B-Spline Curve")
        plt.show()

    else:
        print("No curve detected")
