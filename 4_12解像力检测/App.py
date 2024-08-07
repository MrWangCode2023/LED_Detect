import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


# 设置显示中文字符的字体
def setup_font(font_path="C:/Windows/Fonts/simhei.ttf"):
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

# 读取并预处理图像
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 5)
    return blurred_image

def calculate_and_smooth_center_line(blurred_image, window_size=5):
    center_line = np.mean(blurred_image, axis=0)
    weights = np.ones(window_size) / window_size
    smoothed_center_line = np.convolve(center_line, weights, mode="same")  # 一维卷积
    return smoothed_center_line

def find_max_value_and_thresh(smoothed_center_line):
    max_value = np.max(smoothed_center_line)  # 找到最大值
    max_index = np.argmax(smoothed_center_line)  # 索引

    # thresh@10
    thresh = max_value * 0.1

    # 所有值为thresh的索引
    thresh_indexs = np.where(smoothed_center_line == thresh)[0]

    return max_value, max_index, thresh, thresh_indexs

def visualize_results(original_image, smoothed_center_line, max_value, max_index, thresh, thresh_indexs):
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

# 主函数
def analyze_image(image_path):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("无法读取图像，请检查路径。")
        return

    blurred_image = preprocess_image(original_image)
    smoothed_center_line = calculate_and_smooth_center_line(blurred_image)

    max_value, max_index, thresh, thresh_indexs = find_max_value_and_thresh(smoothed_center_line)

    # 打印返回值
    print(f"最大值： {max_value}")
    print(f"最大值索引： {max_index}")
    print(f"thresh: {thresh}")
    print(f"thresh位置： {thresh_indexs}")

    visualize_results(original_image, smoothed_center_line, max_value, max_index, thresh, thresh_indexs)

if __name__ == "__main__":
    setup_font()
    image_path = "../../../projectData/LED_data/4_12/3.png"
    analyze_image(image_path)
