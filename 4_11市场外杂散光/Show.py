import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk


def round_up_to_nearest_hundred(x):
    return int(np.ceil(x / 100.0)) * 100


def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()


def show_image(image_dict):
    num_images = len(image_dict)
    screen_width, screen_height = get_screen_size()

    # 假设每个子图的大小为 (width, height)
    sub_plot_width_inch = 5  # 每个子图的宽度（英寸）
    sub_plot_height_inch = 5  # 每个子图的高度（英寸）

    # 设置DPI以提高图像清晰度
    dpi = 100

    # 计算每行可以放多少列
    cols = screen_width // (sub_plot_width_inch * dpi)
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * sub_plot_width_inch, rows * sub_plot_height_inch), dpi=dpi)

    # 确保 axes 是一个二维数组
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.reshape(rows, cols)

    # 如果子图比图像多，隐藏多余的子图
    for ax in axes.flat[num_images:]:
        ax.axis('off')

    for ax, (title, image) in zip(axes.flat, image_dict.items()):
        ax.set_title(title, fontsize=14)  # 调整标题字体大小
        if len(image.shape) == 2:  # 灰度图像
            ax.imshow(image, cmap='gray', interpolation='none', extent=[0, image.shape[1], 0, image.shape[0]])
        else:  # 彩色图像
            ax.imshow(image, interpolation='none', extent=[0, image.shape[1], 0, image.shape[0]])

        # 自动获取刻度范围，并取100的倍数
        max_x = round_up_to_nearest_hundred(image.shape[1])
        max_y = round_up_to_nearest_hundred(image.shape[0])

        # 设置刻度范围
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)

        # 显示刻度
        ax.set_xticks(np.arange(0, max_x + 1, step=100))
        ax.set_yticks(np.arange(0, max_y + 1, step=100))

        # 在坐标轴上标出图像尺寸
        ax.set_xlabel(f'W * H: ({image.shape[1]}, {image.shape[0]})PX')

        # 隐藏边框
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=2.0)  # 使用tight_layout优化间距，并设置pad值来减少间隔

    # 使用matplotlib的fullscreen模式
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


# 示例用法
if __name__ == "__main__":
    # 示例图像字典
    example_images = {
        'Image 1': np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8),
        'Image 2': np.random.randint(0, 255, (250, 350, 3), dtype=np.uint8),
        'Image 3': np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
        'Image 4': np.random.randint(0, 255, (150, 250, 3), dtype=np.uint8),
        'Image 5': np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8),
        'Image 6': np.random.randint(0, 255, (350, 450, 3), dtype=np.uint8)
    }
    show_image(example_images)
