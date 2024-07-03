import cv2
import numpy as np
import threading
import queue
import time
from collections import namedtuple
import gxipy as gx
from common.Common import object_extraction, object_curve_fitting, curve_division, draw_rectangle_roi_base_on_points
from common.Common import analyze_image_with_rois

def capture_frames(cam, frame_queue):
    """
    从相机捕获帧并将其放入帧队列中。

    参数：
    cam -- 相机对象，用于捕获图像
    frame_queue -- 帧队列，用于存储捕获的帧
    """
    while True:
        # 从相机的数据流中获取一帧图像
        raw_image = cam.data_stream[0].get_image()

        # 如果图像为空或状态为不完整，继续下一次循环
        if raw_image is None or raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
            continue

        # 将原始图像转换为numpy数组
        numpy_image = raw_image.get_numpy_array()

        # 如果转换后的numpy数组为空，继续下一次循环
        if numpy_image is None:
            continue

        # 获取图像的像素格式
        pixel_format = raw_image.get_pixel_format()

        # 根据像素格式处理图像
        if pixel_format == gx.GxPixelFormatEntry.RGB8:
            # 如果像素格式为RGB8，直接使用numpy数组
            frame = numpy_image
        elif pixel_format == gx.GxPixelFormatEntry.BAYER_RG8:
            # 如果像素格式为BAYER_RG8，将其转换为RGB格式
            frame = cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_RG2RGB)
        else:
            # 其他像素格式不处理，继续下一次循环
            continue

        # 将处理后的帧放入帧队列中
        frame_queue.put(frame)
#################################################################################################


def detect_LED_blinking_from_camera(frame_queue, roi_size):
    roi_status = {}
    start_time = time.time()
    end_time = None

    while True:
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

        if end_time is not None:
            total_time = end_time - start_time
        else:
            total_time = current_time - start_time

        unlit_rois = [i for i in range(1, max(roi_status.keys()) + 1) if roi_status.get(i, 'off') == 'off']

        # 定义每行文本的高度
        line_height = 30

        # 计算每个文本块的行数
        lighted_text_lines = f"Lighted ROI indexes:\n{on_rois}".split('\n')
        unlighted_text_lines = f"Unlighted ROI indexes:\n{unlit_rois}".split('\n')
        total_time_text_lines = f"Total time:\n{total_time:.2f} seconds".split('\n')

        # 绘制点亮的ROI索引
        y_start = 30
        for idx, line in enumerate(lighted_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制未点亮的ROI索引，调整起始位置以避免重叠
        y_start += len(lighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
        for idx, line in enumerate(unlighted_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 绘制总时间，调整起始位置以避免重叠
        y_start += len(unlighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
        for idx, line in enumerate(total_time_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2)

        # 在终端上显示结果
        print(f"Lighted ROIs: {on_rois}")
        print(f"Unlit ROIs: {unlit_rois}")
        print(f"Total time: {total_time:.2f} seconds")

        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#########################################################################################
def camera():
    """
    主函数：初始化相机，启动捕获和检测线程，并处理相机关闭和资源释放。
    """
    try:
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

        # 创建帧队列，设置队列最大容量为100
        frame_queue = queue.Queue(maxsize=100)

        # 启动捕获帧的线程，目标函数为 capture_frames，参数为相机对象和帧队列
        capture_thread = threading.Thread(target=capture_frames, args=(cam, frame_queue))

        # 启动检测LED闪烁的线程，目标函数为 detect_LED_blinking_from_camera，参数为帧队列和ROI大小
        detect_thread = threading.Thread(target=detect_LED_blinking_from_camera, args=(frame_queue, 20))

        # 启动捕获线程
        capture_thread.start()

        # 启动检测线程
        detect_thread.start()

        # 等待捕获线程结束
        capture_thread.join()

        # 等待检测线程结束
        detect_thread.join()

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # 确保相机和资源在退出时正确关闭
        cam.stream_off()
        cam.close_device()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera()


