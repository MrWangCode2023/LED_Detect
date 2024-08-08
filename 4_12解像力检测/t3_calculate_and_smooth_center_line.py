import numpy as np


def t3_ESF(blurred_image, window_size=5):
    average_line = np.mean(blurred_image, axis=0)
    weights = np.ones(window_size) / window_size
    smoothed_average_line = np.convolve(average_line, weights, mode="same")  # 一维卷积
    return average_line, smoothed_average_line
