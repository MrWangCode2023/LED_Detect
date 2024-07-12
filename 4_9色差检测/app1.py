import cv2

from Object_extraction import object_extraction
from Object_color_extraction import object_color_extraction
from RGB_to_CIELAB import rgb_to_cielab
from Color_diference import delta_e_cie2000
from Draw_graph import draw_graph
from Show import show_image




# 标准间-检测件模式
def main1(image1, image2, num=1):
    """
    Args:
        image1: 标准图样本
        image2: 色差检测图

    Returns:
        delta_color: 计算出来的色差值

    """
    # 从左上到右下排序后的contours(区域坐标数组)
    contours1 = object_extraction(image1)
    contours2 = object_extraction(image2)

    num_contours1 = len(contours1)
    num_contours2 = len(contours2)
    count = min(num_contours1, num_contours2)

    # cie_lab1s, cie_lab2s = [], []
    delta_colors = []
    for i in range(count):
        contour1, contour2 = contours1[i], contours2[i]
        rgb1, rgb2 = object_color_extraction(image1, contour1), object_color_extraction(image2, contour2)

        cie_lab1, cie_lab2 = rgb_to_cielab(rgb1), rgb_to_cielab(rgb2)

        delta_color = delta_e_cie2000(cie_lab1, cie_lab2, Kl=1, Kc=1, Kh=1)
        print(f"| point{i} | 色差值:{delta_color} | 标样CIELab值:{cie_lab1} | 试样CIELab值:{cie_lab2} |")
        delta_colors.append(delta_color)

    print(f"\n样本与标样第 {num} 个point的色差值为： {delta_colors[num]}")

    # 绘制图像
    MBox_img1, object_positions1 = draw_graph(image1, contours1)
    MBox_img2, object_positions2 = draw_graph(image2, contours2)
    image_dict = {
        "Image1": image1,
        "MBoxs1": MBox_img1,
        "Image2": image1,
        "MBoxs2": MBox_img2,
    }
    show_image(image_dict)

    return delta_colors


if __name__ == "__main__":
    image1 = cv2.imread("E:\workspace\Data\LED_data\\4_9\\1.png")
    image2 = cv2.imread("E:\workspace\Data\LED_data\\4_9\\1.png")
    main1(image1, image2, 3)