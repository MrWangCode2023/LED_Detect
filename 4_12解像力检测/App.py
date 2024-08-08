import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from t1_setup_font import t1_setup_font
from t2_preprocess_image import t2_preprocess_image
from t3_calculate_and_smooth_center_line import t3_ESF
from t4_find_max_value_and_thresh import t4_peak_resolution_flare
from t5_find_neighbors import t5_find_neighbors
from t6_resolution_calculate import t6_resoluton_calculate
from t7_visualize_results import t7_visualize_results


# 主函数
def app(image, dmp=2, pmr=5):
    t1_setup_font()
    if image is None:
        print("Image is none!")
        return

    show_image = image.copy()

    # 1 preprocess
    blurred_image = t2_preprocess_image(image)

    # 2 ESF
    esf, smoothed_esf = t3_ESF(blurred_image)

    # 3 peak & resolution & flares
    result = t4_peak_resolution_flare(esf)

    # 4 width of flare
    flare_id, flare_width = t5_find_neighbors(result[6], result[1])
    print(f"flare_width: {flare_width}")

    # 5 width of resolution
    reso_id, reso_width = t5_find_neighbors(result[7], result[1])
    print(f"reso_width: {reso_width}")

    # 6 resolution calculation
    resolution = t6_resoluton_calculate(flare_width, dmp, pmr)

    # print
    print(f"resolution: {resolution}")

    # 7 Visualization (image, peak_x, reso_id, flare_id, smoothed_esf)
    t7_visualize_results(show_image, result[1], reso_id, flare_id, smoothed_esf)

    return resolution


if __name__ == "__main__":
    image = cv2.imread("../../../projectData/LED_data/4_12/3.png")
    resolution = app(image)

