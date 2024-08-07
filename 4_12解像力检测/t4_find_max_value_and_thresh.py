import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager



def t4_find_max_value_and_thresh(center_line):
    max_value = np.max(center_line)  # 找到最大值

    max_index = np.argmax(center_line)  # 索引
    # print(f"测试：{center_line}")

    # thresh@10
    thresh = int(max_value * 0.1)

    # 所有值为thresh的索引
    thresh_indexs = np.where((center_line >= thresh) & (center_line <= thresh + 1))[0]


    return max_value, max_index, thresh, thresh_indexs
