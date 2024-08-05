import cv2
import numpy as np
import matplotlib.pyplot as plt
# # from Object_extraction import object_extraction
# from common.Common import show_object_color

def round_up_to_nearest_hundred(x):
    return int(np.ceil(x / 100.0)) * 100

def show_image(image_dict):
    num_images = len(image_dict)
    cols = int(np.ceil(np.sqrt(num_images)))  # 计算列数
    rows = int(np.ceil(num_images / cols))  # 计算行数

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # 动态调整figsize

    # 如果只有一个图像，axes不会是列表，因此将其转换为列表
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()

    # 如果子图比图像多，隐藏多余的子图
    for ax in axes.flat[num_images:]:
        ax.axis('off')

    for ax, (title, image) in zip(axes.flat, image_dict.items()):
        ax.set_title(title, fontsize=14)  # 调整标题字体大小
        if len(image.shape) == 2:  # 灰度图像
            ax.imshow(image, cmap='gray', extent=[0, image.shape[1], 0, image.shape[0]])
        else:  # 彩色图像
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image_rgb, extent=[0, image.shape[1], 0, image.shape[0]])

        # 自动获取刻度范围，并取100的倍数
        max_x = round_up_to_nearest_hundred(image.shape[1])
        max_y = round_up_to_nearest_hundred(image.shape[0])

        # 设置刻度范围
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)  # 设置y轴从下到上

        # 显示刻度
        ax.set_xticks(np.arange(0, max_x + 1, step=100))
        ax.set_yticks(np.arange(0, max_y + 1, step=100))

        # 在坐标轴上标出图像尺寸
        ax.set_xlabel(f'W * H: ({image.shape[1]}, {image.shape[0]})PX')
        # ax.set_ylabel(f'Height: {image.shape[0]} px')

        # 隐藏边框
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=2.0)  # 使用tight_layout优化间距，并设置pad值来减少间隔
    plt.show()



