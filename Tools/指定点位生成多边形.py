import cv2
import numpy as np

def draw_filled_polygon_on_black_image(points, output_path):
    """
    在纯黑图像上绘制实心多边形。

    :param points: 多边形的顶点坐标，格式为 [(x1, y1), (x2, y2), ...]
    :param output_path: 输出图像的路径
    """
    # 创建纯黑图像
    image = np.zeros((960, 960, 3), dtype=np.uint8)

    # 将点转换为numpy数组
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # 在图像上绘制实心多边形
    cv2.fillPoly(image, [pts], color=(255, 255, 255))

    # 保存结果图像
    cv2.imwrite(output_path, image)

    # 显示带有多边形的图像
    cv2.imshow("Filled Polygon Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_path = 'E:\\workspace\\Data\\LED_data\\task4\\32.png'  # 更新为实际的输出图像路径
    points = [(260, 110), (620, 147), (750, 650), (160, 550)]  # 更新为实际的点
    draw_filled_polygon_on_black_image(points, output_path)
