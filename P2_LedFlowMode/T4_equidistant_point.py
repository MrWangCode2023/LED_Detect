import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte


def t4EqualizationPointsAndAngels(image, curve_coordinates, segment_length=18):
    if len(curve_coordinates) < 2:
        raise ValueError("曲线坐标点少于两个，无法计算等分点。")

    point_img = image.copy()
    points_and_angles = []
    accumulated_length = 0.0
    coordinates_num = len(curve_coordinates)

    divided_point = np.array(curve_coordinates[0]).squeeze()
    points_and_angles.append((divided_point, 0))

    for i in range(1, coordinates_num):
        prev_point = np.array(curve_coordinates[i - 1]).squeeze()
        next_point = np.array(curve_coordinates[i]).squeeze()

        if len(prev_point) != 2 or len(next_point) != 2:
            print(f"数据点格式错误: prev_point={prev_point}, next_point={next_point}")
            continue

        distance = np.linalg.norm(next_point - prev_point)
        accumulated_length += distance

        while accumulated_length >= segment_length:
            accumulated_length -= segment_length
            t = 1 - (accumulated_length / distance)
            divided_point = (1 - t) * prev_point + t * next_point

            if len(points_and_angles) == 1:
                angle = np.arctan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) * 180 / np.pi
            else:
                angle = points_and_angles[-1][1]

            points_and_angles.append((divided_point, angle))

            # 在图像上标记等分点
            cv2.circle(point_img, (int(divided_point[0]), int(divided_point[1])), 5, (0, 0, 255), -1)

    for i in range(1, len(points_and_angles) - 1):
        prev_point = points_and_angles[i - 1][0]
        next_point = points_and_angles[i + 1][0]

        if len(prev_point) != 2 or len(next_point) != 2:
            print(f"数据点格式错误: prev_point={prev_point}, next_point={next_point}")
            continue

        tangent_vector = next_point - prev_point
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        angle = np.arctan2(normal_vector[1], normal_vector[0]) * 180 / np.pi
        points_and_angles[i] = (points_and_angles[i][0], angle)

    # 使用 pop(0) 去掉第0个元素
    if points_and_angles:
        points_and_angles.pop(0)  # 确保列表不为空

    # print(f"等分点个个数：{len(points_and_angles)}")

    # print(f"等分点：{[point for point, _ in points_and_angles]}")

    # cv2.imshow("division", point_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return points_and_angles





if __name__ == "__main__":
    from T1_contours import t1Contours
    from T2_thin_contours import t2ThinCurve
    from T3_fitted_curve import t3FittedCurve

    image = cv2.imread(r"E:\workspace\Data\LED_data\\task1\\6.bmp")
    contour_list = t1Contours(image)
    thinned_contour = t2ThinCurve(image, contour_list[0])
    fitted_contour = t3FittedCurve(image, contour_list[0])

    equidistant_points1 = t4EqualizationPointsAndAngels(image, thinned_contour, segment_length=18)
    # print(f"equidistant_points1:{equidistant_points1}")
    equidistant_points2 = t4EqualizationPointsAndAngels(image, fitted_contour, segment_length=18)
    print("正常")
