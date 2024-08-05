import cv2
import numpy as np
from T1_contours import t1Contours
from T2_thin_contours import t2ThinCurve


def t3FittedCurve(image, contour):
    contour_image = image.copy()
    curve_img = image.copy()
    fitted_curve = []

    # 1 绘制轮廓
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 1)

    # 2 轮廓循环曲线拟合
    if len(contour) > 0:
        # 提取轮廓的x和y坐标
        contour = contour.reshape(-1, 2)
        x = contour[:, 0]
        y = contour[:, 1]
        num = len(x)
        # print(f"数量：{num}")

        # 使用多项式拟合曲线，degree可以调整拟合的多项式阶数
        degree = 3
        poly_params = np.polyfit(x, y, degree)
        poly = np.poly1d(poly_params)

        # 绘制拟合的曲线
        x_new = np.linspace(x.min(), x.max(), 50)
        y_new = poly(x_new)

        # 存储拟合后的曲线坐标
        fitted_curve = list(zip(x_new, y_new))
        # curve_coordinate.append(curve_points)

        for i in range(len(x_new) - 1):
            cv2.line(curve_img, (int(x_new[i]), int(y_new[i])), (int(x_new[i + 1]), int(y_new[i + 1])), (255, 0, 0), 1)

    # 打印结果
    # print(f"曲线坐标：{fitted_curve}")

    # 显示结果
    # cv2.imshow('Fitted Curve', curve_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return fitted_curve


if __name__ == "__main__":
    image = cv2.imread(r"E:\workspace\Data\LED_data\\task1\\6.bmp")
    contours = t1Contours(image)
    # thin_curve = thin_contours(image, contours[0])
    fitted_curve = t3FittedCurve(image, contours[0])

############ 2版 #############
# import cv2
# import numpy as np
# from scipy.interpolate import splprep, splev
# from T1_contours import contours
# from T2_thin_contours import thin_contours
#
# def contour2curve(image, contours):
#     contour_image = image.copy()
#     curve_img = image.copy()
#     curve_coordinate = []
#
#     # 1. 绘制轮廓
#     cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
#
#     # 2. 轮廓循环曲线拟合
#     for contour in contours:
#         if len(contour) > 0:
#             # 提取轮廓的x和y坐标
#             contour = contour.reshape(-1, 2)
#             x = contour[:, 0]
#             y = contour[:, 1]
#
#             # 使用样条插值拟合曲线
#             tck, u = splprep([x, y], s=0)
#             x_new, y_new = splev(np.linspace(0, 1, 30), tck)
#
#             # 存储拟合后的曲线坐标
#             curve_points = list(zip(x_new, y_new))
#             curve_coordinate.append(curve_points)
#
#             for i in range(len(x_new) - 1):
#                 cv2.line(curve_img, (int(x_new[i]), int(y_new[i])), (int(x_new[i + 1]), int(y_new[i + 1])), (255, 0, 0), 1)
#
#     # 打印结果
#     print(f"曲线坐标：{curve_coordinate}")
#
#     # 显示结果
#     cv2.imshow('contour_image', contour_image)
#     cv2.imshow('Fitted Curve', curve_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     return curve_coordinate
#
# if __name__ == "__main__":
#     image = cv2.imread(r"E:\workspace\Data\LED_data\\task1\\3.bmp")
#     contours = contours(image)
#     thinned_contours = thin_contours(image, contours)
#     fitted_curves = contour2curve(image, thinned_contours)

#################### 3版 ###################
# import cv2
# import numpy as np
# from scipy.interpolate import splprep, splev
# from T1_contours import contours
# from T2_thin_contours import thin_contours
#
# def contour2curve(image, contours):
#     contour_image = image.copy()
#     curve_img = image.copy()
#     curve_coordinate = []
#
#     # 1. 绘制轮廓
#     cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
#
#     # 2. 轮廓循环曲线拟合
#     for contour in contours:
#         if len(contour) > 0:
#             # 提取轮廓的x和y坐标
#             contour = contour.reshape(-1, 2)
#             x = contour[:, 0]
#             y = contour[:, 1]
#
#             # 使用B样条插值拟合曲线
#             tck, u = splprep([x, y], s=0)
#             x_new, y_new = splev(np.linspace(0, 1, 1000), tck)
#
#             # 存储拟合后的曲线坐标
#             curve_points = list(zip(x_new, y_new))
#             curve_coordinate.append(curve_points)
#
#             for i in range(len(x_new) - 1):
#                 cv2.line(curve_img, (int(x_new[i]), int(y_new[i])), (int(x_new[i + 1]), int(y_new[i + 1])), (255, 0, 0), 2)
#
#     # 打印结果
#     print(f"曲线坐标：{curve_coordinate}")
#
#     # 显示结果
#     cv2.imshow('contour_image', contour_image)
#     cv2.imshow('Fitted Curve', curve_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     return curve_coordinate
#
# if __name__ == "__main__":
#     image = cv2.imread(r"E:\workspace\Data\LED_data\\task1\\3.bmp")
#     contours = contours(image)
#     thinned_contours = thin_contours(image, contours)
#     fitted_curves = contour2curve(image, thinned_contours)

