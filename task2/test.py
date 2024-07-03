import cv2
import numpy as np
import threading
import queue
import time
import gxipy as gx
from common.Common import analyze_image_with_rois

# 定义一个函数来捕获相机帧
def capture_frames(cam, frame_queue):
    while True:
        # 获取相机图像
        raw_image = cam.data_stream[0].get_image()
        if raw_image is None or raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
            continue

        # 将相机图像转换为numpy数组
        numpy_image = raw_image.get_numpy_array()
        if numpy_image is None:
            continue

        # 检查图像格式并转换
        pixel_format = raw_image.get_pixel_format()
        if pixel_format == gx.GxPixelFormatEntry.RGB8:
            frame = numpy_image
        elif pixel_format == gx.GxPixelFormatEntry.BAYER_RG8:
            frame = cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_RG2RGB)
        else:
            continue

        # 将帧放入队列
        frame_queue.put(frame)

# 定义一个函数来检测LED闪烁
def detect_LED_blinking_from_camera(frame_queue, roi_size):
    roi_status = {}
    start_time = time.time()
    end_time = None

    while True:
        # 从队列中获取帧
        frame = frame_queue.get()
        _, analysis_results = analyze_image_with_rois(frame, roi_size=roi_size)

        current_time = time.time()
        on_rois = []
        unlit_rois = []

        for result in analysis_results:
            roi_id = result['roi_id']
            mean_brightness = result['mean_brightness']

            if mean_brightness > 50:
                on_rois.append(roi_id)
                if roi_id not in roi_status:
                    roi_status[roi_id] = 'on'
                    if all([roi_status.get(i, 'off') == 'on' for i in range(1, roi_id)]):
                        end_time = time.time()
                    else:
                        print(f"Error in lighting sequence at ROI {roi_id}")
            else:
                unlit_rois.append(roi_id)
                if roi_id in roi_status and roi_status[roi_id] == 'on':
                    print(f"Error: ROI {roi_id} turned off")

        all_on = all(status == 'on' for status in roi_status.values())
        if all_on:
            end_time = time.time()

        total_time = (end_time or current_time) - start_time

        unlit_rois = [i for i in range(1, max(roi_status.keys()) + 1) if roi_status.get(i, 'off') == 'off']

        # 在图像上显示结果
        display_text(frame, f"Lighted ROI indexes:\n{on_rois}", (10, 30), (0, 255, 0))
        display_text(frame, f"Unlighted ROI indexes:\n{unlit_rois}", (10, 30 + len(on_rois) * 30 + 10), (0, 0, 255))
        display_text(frame, f"Total time:\n{total_time:.2f} seconds", (10, 30 + (len(on_rois) + len(unlit_rois)) * 30 + 20), (255, 255, 255))

        # 在终端上显示结果
        print(f"Lighted ROIs: {on_rois}")
        print(f"Unlit ROIs: {unlit_rois}")
        print(f"Total time: {total_time:.2f} seconds")

        # 显示图像
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 定义一个辅助函数来在图像上显示多行文本
def display_text(frame, text, position, color):
    lines = text.split('\n')
    line_height = 30
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (position[0], position[1] + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 定义主函数
def main():
    """
    主函数：初始化相机，启动捕获和检测线程，并处理相机关闭和资源释放。
    """
    # 初始化相机API，获取设备管理器实例
    device_manager = gx.DeviceManager()

    # 更新设备列表，获取设备数量和设备信息列表
    dev_num, dev_info_list = device_manager.update_device_list()

    # 检查是否检测到设备
    if dev_num == 0:
        print("没有检测到设备")
        return

    # 打开相机，使用设备信息列表中的第一个设备的序列号
    cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))

    # 关闭触发模式，设置为连续采集模式
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    # 开始相机数据流
    cam.stream_on()

    # 创建帧队列，设置队列最大容量为1000
    frame_queue = queue.Queue(maxsize=1000)

    # 启动捕获帧的线程，目标函数为capture_frames，参数为相机对象和帧队列
    capture_thread = threading.Thread(target=capture_frames, args=(cam, frame_queue))

    # 启动检测LED闪烁的线程，目标函数为detect_LED_blinking_from_camera，参数为帧队列和ROI大小
    detect_thread = threading.Thread(target=detect_LED_blinking_from_camera, args=(frame_queue, 15))

    # 启动捕获线程
    capture_thread.start()

    # 启动检测线程
    detect_thread.start()

    # 等待捕获线程结束
    capture_thread.join()

    # 等待检测线程结束
    detect_thread.join()

    # 停止相机数据流
    cam.stream_off()

    # 关闭相机设备
    cam.close_device()

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
