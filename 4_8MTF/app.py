import cv2
import matplotlib.pyplot as plt
from Box import bbox, mbbox
from MTF import mtf, esf, lsf



def app(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    b_line1_pixel_values, b_line2_pixel_values = bbox(contours, image)
    m_line1_pixel_values, m_line2_pixel_values = mbbox(contours, image)

    for i in range(len(b_line1_pixel_values)):
        b_esf1 = esf(b_line1_pixel_values[i])
        b_esf2 = esf(b_line2_pixel_values[i])
        m_esf1 = esf(m_line1_pixel_values[i])
        m_esf2 = esf(m_line2_pixel_values[i])

        b_lsf1 = lsf(b_esf1)
        b_lsf2 = lsf(b_esf2)
        m_lsf1 = lsf(m_esf1)
        m_lsf2 = lsf(m_esf2)

        b_mtf1 = mtf(b_lsf1)
        b_mtf2 = mtf(b_lsf2)
        m_mtf1 = mtf(m_lsf1)
        m_mtf2 = mtf(m_lsf2)

        # b_esf1_smooth = smooth_data(b_esf1)
        # b_esf2_smooth = smooth_data(b_esf2)
        # m_esf1_smooth = smooth_data(m_esf1)
        # m_esf2_smooth = smooth_data(m_esf2)
        #
        # b_lsf1_smooth = smooth_data(b_lsf1)
        # b_lsf2_smooth = smooth_data(b_lsf2)
        # m_lsf1_smooth = smooth_data(m_lsf1)
        # m_lsf2_smooth = smooth_data(m_lsf2)
        #
        # b_mtf1_smooth = smooth_data(b_mtf1)
        # b_mtf2_smooth = smooth_data(b_mtf2)
        # m_mtf1_smooth = smooth_data(m_mtf1)
        # m_mtf2_smooth = smooth_data(m_mtf2)

        plt.figure(figsize=(20, 10))

        # ESF
        plt.subplot(3, 2, 1)
        plt.plot(b_esf1, label='Arc Direction')
        plt.plot(b_esf2, label='Sagittal Direction')
        plt.title(f'Bounding Box ESF for Object {i+1}')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(m_esf1, label='Arc Direction')
        plt.plot(m_esf2, label='Sagittal Direction')
        plt.title(f'Min Area Box ESF for Object {i+1}')
        plt.legend()

        # LSF
        plt.subplot(3, 2, 3)
        plt.plot(b_lsf1, label='Arc Direction')
        plt.plot(b_lsf2, label='Sagittal Direction')
        plt.title(f'Bounding Box LSF for Object {i+1}')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(m_lsf1, label='Arc Direction')
        plt.plot(m_lsf2, label='Sagittal Direction')
        plt.title(f'Min Area Box LSF for Object {i+1}')
        plt.legend()

        # MTF
        plt.subplot(3, 2, 5)
        plt.plot(b_mtf1, label='Arc Direction')
        plt.plot(b_mtf2, label='Sagittal Direction')
        plt.title(f'Bounding Box MTF for Object {i+1}')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(m_mtf1, label='Arc Direction')
        plt.plot(m_mtf2, label='Sagittal Direction')
        plt.title(f'Min Area Box MTF for Object {i+1}')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\34.png")
    app(image)