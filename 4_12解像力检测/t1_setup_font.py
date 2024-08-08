import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


# 设置显示中文字符的字体
def t1_setup_font(font_path="C:/Windows/Fonts/simhei.ttf"):
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False