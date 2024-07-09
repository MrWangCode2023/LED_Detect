import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Mid_point import mid_point


def bbox(contours, image):
    line1_pixel_values = []
    line2_pixel_values = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.int32)

        a, b, c, d = box[0], box[1], box[2], box[3]
        mid_ab = np.intp(mid_point(a, b))
        mid_cd = np.intp(mid_point(c, d))
        mid_bc = np.intp(mid_point(b, c))
        mid_da = np.intp(mid_point(d, a))

        mask_line1 = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_line2 = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.line(mask_line1, pt1=mid_ab, pt2=mid_cd, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)
        cv2.line(mask_line2, pt1=mid_bc, pt2=mid_da, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)

        line1_pixels = image[mask_line1 == 255]
        line2_pixels = image[mask_line2 == 255]

        line1_pixel_values.append(line1_pixels)
        line2_pixel_values.append(line2_pixels)

    return line1_pixel_values, line2_pixel_values


def mbbox(contours, image):
    line1_pixel_values = []
    line2_pixel_values = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        a, b, c, d = box[0], box[1], box[2], box[3]
        mid_ab = np.intp(mid_point(a, b))
        mid_bc = np.intp(mid_point(b, c))
        mid_cd = np.intp(mid_point(c, d))
        mid_da = np.intp(mid_point(d, a))

        mask_line1 = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_line2 = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.line(mask_line1, pt1=mid_ab, pt2=mid_cd, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)
        cv2.line(mask_line2, pt1=mid_bc, pt2=mid_da, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)

        line1_pixels = image[mask_line1 == 255]
        line2_pixels = image[mask_line2 == 255]

        line1_pixel_values.append(line1_pixels)
        line2_pixel_values.append(line2_pixels)

    return line1_pixel_values, line2_pixel_values


def calculate_esf(pixel_values):
    esf = pixel_values
    return esf


def calculate_lsf(esf):
    lsf = np.diff(esf, axis=0)
    return lsf


def calculate_mtf(lsf):
    mtf = np.abs(np.fft.fft(lsf, axis=0))
    mtf = mtf / np.max(mtf)
    return mtf


def smooth_data(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma)


def app(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    b_line1_pixel_values, b_line2_pixel_values = bbox(contours, image)
    m_line1_pixel_values, m_line2_pixel_values = mbbox(contours, image)

    for i in range(len(b_line1_pixel_values)):
        b_esf1 = calculate_esf(b_line1_pixel_values[i])
        b_esf2 = calculate_esf(b_line2_pixel_values[i])
        m_esf1 = calculate_esf(m_line1_pixel_values[i])
        m_esf2 = calculate_esf(m_line2_pixel_values[i])

        b_lsf1 = calculate_lsf(b_esf1)
        b_lsf2 = calculate_lsf(b_esf2)
        m_lsf1 = calculate_lsf(m_esf1)
        m_lsf2 = calculate_lsf(m_esf2)

        b_mtf1 = calculate_mtf(b_lsf1)
        b_mtf2 = calculate_mtf(b_lsf2)
        m_mtf1 = calculate_mtf(m_lsf1)
        m_mtf2 = calculate_mtf(m_lsf2)

        b_esf1_smooth = smooth_data(b_esf1)
        b_esf2_smooth = smooth_data(b_esf2)
        m_esf1_smooth = smooth_data(m_esf1)
        m_esf2_smooth = smooth_data(m_esf2)

        b_lsf1_smooth = smooth_data(b_lsf1)
        b_lsf2_smooth = smooth_data(b_lsf2)
        m_lsf1_smooth = smooth_data(m_lsf1)
        m_lsf2_smooth = smooth_data(m_lsf2)

        b_mtf1_smooth = smooth_data(b_mtf1)
        b_mtf2_smooth = smooth_data(b_mtf2)
        m_mtf1_smooth = smooth_data(m_mtf1)
        m_mtf2_smooth = smooth_data(m_mtf2)

        plt.figure(figsize=(20, 10))

        # ESF
        plt.subplot(3, 2, 1)
        plt.plot(b_esf1_smooth, label='Arc Direction')
        plt.plot(b_esf2_smooth, label='Sagittal Direction')
        plt.title(f'Bounding Box ESF for Object {i+1}')
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(m_esf1_smooth, label='Arc Direction')
        plt.plot(m_esf2_smooth, label='Sagittal Direction')
        plt.title(f'Min Area Box ESF for Object {i+1}')
        plt.legend()

        # LSF
        plt.subplot(3, 2, 3)
        plt.plot(b_lsf1_smooth, label='Arc Direction')
        plt.plot(b_lsf2_smooth, label='Sagittal Direction')
        plt.title(f'Bounding Box LSF for Object {i+1}')
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(m_lsf1_smooth, label='Arc Direction')
        plt.plot(m_lsf2_smooth, label='Sagittal Direction')
        plt.title(f'Min Area Box LSF for Object {i+1}')
        plt.legend()

        # MTF
        plt.subplot(3, 2, 5)
        plt.plot(b_mtf1_smooth, label='Arc Direction')
        plt.plot(b_mtf2_smooth, label='Sagittal Direction')
        plt.title(f'Bounding Box MTF for Object {i+1}')
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(m_mtf1_smooth, label='Arc Direction')
        plt.plot(m_mtf2_smooth, label='Sagittal Direction')
        plt.title(f'Min Area Box MTF for Object {i+1}')
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\34.png")
    app(image)
