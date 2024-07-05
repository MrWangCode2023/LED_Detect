import cv2

def invert_image(image_path, output_path):
    # 加载图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("错误：无法加载图像")
        return

    # 反转图像
    inverted_image = cv2.bitwise_not(image)

    # 保存反转后的图像
    cv2.imwrite(output_path, inverted_image)

    # 显示原始图像和反转后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Inverted Image", inverted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'E:\workspace\Data\LED_data\\task4\\22.png'  # 更新为实际的图像路径
    output_path = 'E:\workspace\Data\LED_data\\task4\\27.png'  # 更新为实际的输出图像路径
    invert_image(image_path, output_path)
