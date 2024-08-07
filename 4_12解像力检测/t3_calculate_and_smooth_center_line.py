import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


def t3_calculate_and_smooth_line(blurred_image, window_size=5):
    center_line = np.mean(blurred_image, axis=0)
    weights = np.ones(window_size) / window_size
    smoothed_center_line = np.convolve(center_line, weights, mode="same")  # 一维卷积
    return center_line, smoothed_center_line
