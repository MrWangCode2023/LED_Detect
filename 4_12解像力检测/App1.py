import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from t1_setup_font import t1_setup_font
from t2_preprocess_image import t2_preprocess_image
from t3_calculate_and_smooth_center_line import t3_calculate_and_smooth_line
from t4_find_max_value_and_thresh import t4_find_max_value_and_thresh
from t5_visualize_results import t5_visualize_results





# 主函数
def app(original_image):
    t1_setup_font()
    if original_image is None:
        print("无法读取图像，请检查路径。")
        return

    # 高斯降噪
    blurred_image = t2_preprocess_image(original_image)

    #
    center_line, smoothed_line = t3_calculate_and_smooth_line(blurred_image)

    max_value, max_index, thresh, thresh_indexs = t4_find_max_value_and_thresh(center_line)

    # 打印返回值
    print(f"最大值： {max_value}")
    print(f"最大值索引： {max_index}")
    print(f"thresh: {thresh}")
    print(f"thresh位置： {thresh_indexs}")

    t5_visualize_results(original_image, smoothed_line, max_value, max_index, thresh, thresh_indexs)

if __name__ == "__main__":
    image = cv2.imread("../../../projectData/LED_data/4_12/3.png")
    app(image)

