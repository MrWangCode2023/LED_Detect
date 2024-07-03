import cv2
import numpy as np


def img_to_luminance(image):
    """
    计算图像的亮度
    :param image: 输入图像 (RGB)
    :return: 亮度图像
    """
    # 将图像转换为浮点数类型
    # image = image.astype(np.float32) / 255
    #
    # # 分离RGB通道
    # B, G, R = cv2.split(image)
    #
    # # 计算亮度
    # luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B

    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # print("luminance", luminance)

    # cv2.imshow("1", luminance)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return luminance

if __name__ == "__main__":
    image_path = r"E:\workspace\Data\LED_data\task4\2.bmp"
    image = cv2.imread(image_path)
    luminance = img_to_luminance(image)
