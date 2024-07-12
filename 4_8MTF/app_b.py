import cv2
import numpy as np
import matplotlib.pyplot as plt
from Box import bbox, mbbox
from MTF import mtf, esf, lsf
from Object_extraction import object_extraction
from MTF_evaluation import mtf_evaluation


def app(image, num):
    # 检查图像是否加载成功
    if image is None:
        print("Error: Image not found or cannot be read.")
        return

    contours = object_extraction(image)

    # 检查是否找到了轮廓
    if not contours:
        print("No contours found in the image.")
        return

    try:
        # 获取像素值
        b_line1_pixel_values, b_line2_pixel_values = bbox(contours, image)

        # 检查索引是否在范围内
        if num <= 0 or num > len(b_line1_pixel_values):
            print(f"Invalid object number: {num}. Must be between 1 and {len(b_line1_pixel_values)}.")
            return

        i = num - 1
        b_esf1 = esf(b_line1_pixel_values[i])
        b_esf2 = esf(b_line2_pixel_values[i])

        b_lsf1 = lsf(b_esf1)
        b_lsf2 = lsf(b_esf2)

        b_mtf1 = mtf(b_lsf1)
        b_mtf2 = mtf(b_lsf2)

        # MTF评价
        b_mtf_eval1 = mtf_evaluation(b_mtf1)
        b_mtf_eval2 = mtf_evaluation(b_mtf2)

        plt.figure(figsize=(20, 10))

        # ESF
        plt.subplot(3, 1, 1)
        plt.plot(b_esf1, label='Arc Direction')
        plt.plot(b_esf2, label='Sagittal Direction')
        plt.title(f'Bounding Box ESF for Object {num}')
        plt.legend()

        # LSF
        plt.subplot(3, 1, 2)
        plt.plot(b_lsf1, label='Arc Direction')
        plt.plot(b_lsf2, label='Sagittal Direction')
        plt.title(f'Bounding Box LSF for Object {num}')
        plt.legend()

        # MTF
        plt.subplot(3, 1, 3)
        plt.plot(b_mtf1, label='Arc Direction')
        plt.plot(b_mtf2, label='Sagittal Direction')
        plt.title(f'Bounding Box MTF for Object {num}')
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"\n第 {num} 个 BBox 的 MTF 评价：\n弧矢方向：\n{b_mtf_eval1}\n子午方向：\n{b_mtf_eval2}\n")

if __name__ == "__main__":
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\33.png")
    app(image, 10)
