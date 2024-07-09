import cv2
import matplotlib.pyplot as plt
from Box import mbbox
from MTF import mtf, esf, lsf
from Object_extraction import object_extraction


def app_m(image, num=1):
    contours = object_extraction(image)

    m_line1_pixel_values, m_line2_pixel_values = mbbox(contours, image)

    if len(m_line1_pixel_values):
        # ESF计算
        i = num - 1
        m_esf1 = esf(m_line1_pixel_values[i])
        m_esf2 = esf(m_line2_pixel_values[i])

        m_lsf1 = lsf(m_esf1)
        m_lsf2 = lsf(m_esf2)

        m_mtf1 = mtf(m_lsf1)
        m_mtf2 = mtf(m_lsf2)

        plt.figure(figsize=(20, 10))

        # ESF
        plt.subplot(3, 1, 1)
        plt.plot(m_esf1, label='Arc Direction')
        plt.plot(m_esf2, label='Sagittal Direction')
        plt.title(f'Min Area Box ESF for Object {i+1}')
        plt.legend()

        # LSF
        plt.subplot(3, 1, 2)
        plt.plot(m_lsf1, label='Arc Direction')
        plt.plot(m_lsf2, label='Sagittal Direction')
        plt.title(f'Min Area Box LSF for Object {i+1}')
        plt.legend()

        # MTF
        plt.subplot(3, 1, 3)
        plt.plot(m_mtf1, label='Arc Direction')
        plt.plot(m_mtf2, label='Sagittal Direction')
        plt.title(f'Min Area Box MTF for Object {i+1}')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\33.png")
    app_m(image, num=2)