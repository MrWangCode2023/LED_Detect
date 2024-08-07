import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


# 读取并预处理图像
def t2_preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 5)
    return blurred_image
