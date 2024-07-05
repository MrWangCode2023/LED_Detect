import cv2
import numpy as np

def draw_bounding_quadrilateral(image_path):
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法加载图像")
        return

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓并绘制外接四边形
    for contour in contours:
        # 使用多边形逼近轮廓
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果近似轮廓有4个顶点，则它是一个四边形
        if len(approx) == 4:
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 1)

    # 保存结果图像
    # cv2.imwrite(output_path, image)

    # 显示原始图像和带有外接四边形的图像
    cv2.imshow("Bounding Quadrilateral", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'E:\\workspace\\Data\\LED_data\\task4\\32.png'  # 更新为实际的图像路径
    # output_path = 'output_image.jpg'  # 更新为实际的输出图像路径
    draw_bounding_quadrilateral(image_path)
