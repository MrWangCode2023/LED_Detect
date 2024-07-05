import cv2
import numpy as np

def rotate_image(image_path, angle, output_path):
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print("错误：无法加载图像")
        return

    # 获取图像的尺寸
    (h, w) = image.shape[:2]

    # 计算图像中心
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 执行仿射变换（旋转图像）
    rotated = cv2.warpAffine(image, M, (w, h))

    # 保存结果图像
    cv2.imwrite(output_path, rotated)

    # 显示原始图像和旋转后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Rotated Image", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'E:\\workspace\\Data\\LED_data\\task4\\30.png'  # 更新为实际的图像路径
    output_path = 'E:\\workspace\\Data\\LED_data\\task4\\31.png'  # 更新为实际的输出图像路径
    angle = 10  # 指定旋转角度
    rotate_image(image_path, angle, output_path)
