import cv2
import numpy as np
import matplotlib.pyplot as plt
from Object_extraction import object_extraction
from Show import show_image

def point_cloud_segmentation(image, k=4):
    points, image_dict = object_extraction(image)
    image_with_neighbors = image.copy()
    points_mesh_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # 单通道灰度图像

    nearest_neighbors = []
    for i, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        nearest = np.argsort(distances)[1:k + 1]  # 排除自身，选择最近的k个点
        nearest_neighbors.append(nearest)

    # 绘制每个点与其最近k个邻居的连线
    for i, neighbors in enumerate(nearest_neighbors):
        for neighbor in neighbors:
            pt1 = tuple(points[i])
            pt2 = tuple(points[neighbor])
            cv2.line(image_with_neighbors, pt1, pt2, (255, 255, 255), 1)  # 使用白色连线
            cv2.line(points_mesh_image, pt1, pt2, 255, 1)  # 使用白色连线

    # 绘制每个点
    for point in points:
        cv2.circle(image_with_neighbors, tuple(point), 3, (0, 0, 255), -1)  # 使用红色绘制点

    # 获取网格填充的图像
    point_segment_image = points_mesh_image.copy()
    # 找到所有的轮廓
    contours, hierarchy = cv2.findContours(point_segment_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 填充所有轮廓
    cv2.drawContours(point_segment_image, contours, -1, 255, thickness=cv2.FILLED)

    seg_image = cv2.cvtColor(point_segment_image, cv2.COLOR_GRAY2BGR)

    image_dict.update({
        # 'image_with_neighbors': image_with_neighbors,
        # "Points_mesh_image": points_mesh_image,
        "Points_segment_image": seg_image
                       })
    show_image(image_dict)

    # seg_image = cv2.cvtColor(point_segment_image, cv2.COLOR_GRAY2BGR)

    # 保存图像
    # cv2.imwrite("E:\\workspace\\Data\\LED_data\\task4\\19.png", point_segment_image)

    return image_dict

if __name__ == "__main__":
    # 读取图像
    # E:\\workspace\\Data\\LED_data\\task4\\17.png
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\17.png")
    # 调用函数并显示结果
    point_segment_image = point_cloud_segmentation(image, k=4)

