import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=1):
    """
    从视频中提取帧并保存为图像。

    Args:
        video_path: 视频文件的路径。
        output_folder: 输出文件夹的路径，用于保存提取的帧。
        frame_interval: 提取帧的间隔，默认为1，表示每一帧都提取。
    """
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 确保视频文件成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读取完毕，退出循环

        # 根据帧间隔决定是否保存当前帧
        if frame_count % frame_interval == 0:
            # 构造保存图像的文件名
            output_file = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.png")
            cv2.imwrite(output_file, frame)  # 保存当前帧
            saved_frame_count += 1

        frame_count += 1

    cap.release()  # 释放视频捕获对象
    print(f"提取完成，已保存 {saved_frame_count} 帧到 '{output_folder}' 文件夹")

# 示例用法
video_path = r"E:\workspace\Data\LED_data\task2\4.avi"  # 视频文件路径
output_folder = r"E:\workspace\Data\LED_data\task2\extracted_frames"  # 输出文件夹路径
extract_frames(video_path, output_folder, frame_interval=10)  # 每10帧提取1帧
