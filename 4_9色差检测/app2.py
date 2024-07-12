import cv2

from Object_extraction import object_extraction
from Object_color_extraction import object_color_extraction
from RGB_to_CIELAB import rgb_to_cielab
from Color_diference import delta_e_cie2000
from Draw_graph import draw_graph
from Show import show_image



# 自检模式
def main2(image1, num1=1, num2=2):
    """
    Args:
        image1: 色差检测图
        num1: 点位
        num2: 点位

    Returns:
        delta_color: 计算出来的色差值

    """
    # 从左上到右下排序后的contours(区域坐标数组)
    contours1 = object_extraction(image1)

    count = len(contours1)

    cie_labs1 = []
    # delta_colors = []
    for i in range(count):
        contour1 = contours1[i]
        rgb1 = object_color_extraction(image1, contour1)
        cie_lab1 = rgb_to_cielab(rgb1)
        print(f"| point{i} | 试样CIELab值：{cie_lab1} |")
        cie_labs1.append(cie_lab1)

    delta_color = delta_e_cie2000(cie_labs1[num1], cie_labs1[num2], Kl=1, Kc=1, Kh=1)

    print(f"当前样本检测到 {count} 个point\n\n样本第 {num1} 和 {num2} 个point的色差值为： {delta_color}")

    # 绘制图像
    MBox_img1, object_positions1 = draw_graph(image1, contours1)
    image_dict = {
        "Origin": image1,
        "MBox": MBox_img1,
    }
    show_image(image_dict)

    return delta_color


if __name__ == "__main__":
    image1 = cv2.imread("E:\workspace\Data\LED_data\\4_9\\1.png")
    delta_color = main2(image1, num1=2, num2=2)