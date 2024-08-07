import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager



def t5_visualize_results(original_image, smoothed_center_line, max_value, max_index, thresh, thresh_indexs):
    output_image = original_image.copy()

    # 标记最大值
    cv2.circle(output_image, (max_index, output_image.shape[0] // 2), 1, (0, 255, 0), -1)  # 最大值位置

    # 标记thresh值的所有位置
    for position in thresh_indexs:
        cv2.circle(output_image, (position, output_image.shape[0] // 2), 5, (255, 0, 0), -1)  # thresh值的位置

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("原始图像")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("平滑中心线像素值")
    plt.plot(smoothed_center_line, color="red", label="平滑中心线")
    plt.axvline(x=max_index, color="green", linestyle="--", label="最大值")  # 标记最大值
    for position in thresh_indexs:
        plt.axvline(x=position, color="blue", linestyle="--", label="thresh")  # 标记thresh
    plt.xlabel("像素位置")
    plt.ylabel("像素值")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("波峰宽度")
    plt.bar([0, 1], [max_value, thresh], color=["green", "blue"], tick_label=["最大值", "thresh"])
    plt.ylabel("像素值")

    plt.tight_layout()
    plt.show()

