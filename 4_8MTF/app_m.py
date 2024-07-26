import cv2
import matplotlib.pyplot as plt
from Box import mbbox
from MTF import mtf, esf, lsf
from Object_extraction import object_extraction
from MTF_evaluation import mtf_evaluation


def app_m(image, num=1):
    # 检查图像是否加载成功
    if image is None:
        print("Error: Image not found or cannot be read.")
        return

    # 1 检测到区域
    contours = object_extraction(image)

    # 检查是否找到了轮廓
    if not contours:
        print("No contours found in the image.")
        return

    try:
        # 获取各个区域弧矢线和子午线像素值
        m_line1_pixel_values, m_line2_pixel_values = mbbox(contours, image)

        # 检查索引是否在范围内
        if num <= 0 or num > len(m_line1_pixel_values):
            print(f"Invalid object number: {num}. Must be between 1 and {len(m_line1_pixel_values)}.")
            return

        # ESF计算
        i = num - 1
        m_esf1 = esf(m_line1_pixel_values[i])
        m_esf2 = esf(m_line2_pixel_values[i])

        m_lsf1 = lsf(m_esf1)
        m_lsf2 = lsf(m_esf2)

        m_mtf1 = mtf(m_lsf1)
        m_mtf2 = mtf(m_lsf2)

        # MTF评价
        m_mtf_eval1 = mtf_evaluation(m_mtf1)
        m_mtf_eval2 = mtf_evaluation(m_mtf2)

        plt.figure(figsize=(20, 10))

        # ESF
        plt.subplot(3, 1, 1)
        plt.plot(m_esf1, label='Arc Direction')
        plt.plot(m_esf2, label='Sagittal Direction')
        plt.title(f'Min Area Box ESF for Object {num}')
        plt.legend()

        # LSF
        plt.subplot(3, 1, 2)
        plt.plot(m_lsf1, label='Arc Direction')
        plt.plot(m_lsf2, label='Sagittal Direction')
        plt.title(f'Min Area Box LSF for Object {num}')
        plt.legend()

        # MTF
        plt.subplot(3, 1, 3)
        plt.plot(m_mtf1, label='Arc Direction')
        plt.plot(m_mtf2, label='Sagittal Direction')
        plt.title(f'Min Area Box MTF for Object {num}')
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"\n第 {num} 个 MBox 的 MTF 评价：\n弧矢方向：\n{m_mtf_eval1}\n子午方向：\n{m_mtf_eval2}")


if __name__ == "__main__":
    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\33.png")
    app_m(image, num=2)