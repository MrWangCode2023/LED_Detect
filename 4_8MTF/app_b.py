import cv2
import matplotlib.pyplot as plt
from Box import bbox
from MTF import mtf, esf, lsf
from Object_extraction import object_extraction


def app_p(image, num=1):
    esf_values = []
    lsf_values = []
    mtf_values = []
    contours = object_extraction(image)

    b_line1_pixel_values, b_line2_pixel_values = bbox(contours, image)

    for i in range(len(b_line1_pixel_values)):
    # if len(b_line1_pixel_values):
        # ESF计算
        # i = num -1
        b_esf1 = esf(b_line1_pixel_values[i])
        b_esf2 = esf(b_line2_pixel_values[i])
        esf_values.append((b_esf1, b_esf2))

        # LSF计算
        b_lsf1 = lsf(b_esf1)
        b_lsf2 = lsf(b_esf2)
        lsf_values.append((b_lsf1, b_lsf2))

        # MTF计算
        b_mtf1 = mtf(b_lsf1)
        b_mtf2 = mtf(b_lsf2)
        mtf_values.append((b_mtf1, b_mtf2))

    if len(b_line1_pixel_values):
        plt.figure(figsize=(20, 10))

        # 清除当前子图
        plt.subplot(3, 1, 1).clear()
        b_esf = esf_values[num - 1]
        b_esf1, b_esf2 = b_esf[0], b_esf[1]
        plt.plot(b_esf1,
                 label='Arc Direction')
        plt.plot(b_esf2,
                 label='Sagittal Direction')
        plt.title(f'Bounding Box ESF for Object {i + 1}')
        plt.legend()

        plt.subplot(3, 1, 2).clear()
        b_lsf = lsf_values[num - 1]
        b_lsf1, b_lsf2 = b_lsf[0], b_lsf[1]
        plt.plot(b_lsf1,
                 label='Arc Direction'
                 )
        plt.plot(b_lsf2,
                 label='Sagittal Direction')
        plt.title(f'Bounding Box LSF for Object {i + 1}')
        plt.legend()

        plt.subplot(3, 1, 3).clear()
        b_mtf = lsf_values[num - 1]
        b_mtf1, b_mtf2 = b_mtf[0], b_mtf[1]
        plt.plot(b_mtf1,
                 label='Arc Direction'
                 )
        plt.plot(b_mtf2,
                 label='Sagittal Direction'
                 )
        plt.title(f'Bounding Box MTF for Object {i + 1}')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

    result = mtf_values[num-1]

    print(f"第 {num} 个区域的MTF值为：\n{result[0]}\n{result[1]}")

    return mtf_values


if __name__ == "__main__":
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\33.png")
    app_p(image, 3)
