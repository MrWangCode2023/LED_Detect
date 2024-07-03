# import cv2
# import numpy as np
# import threading
# import queue
# import time
# from collections import namedtuple
# import gxipy as gx
# from common.Common import object_extraction, object_curve_fitting, curve_division, draw_rectangle_roi_base_on_points
# from common.Common import analyze_image_with_rois
#
#
# def capture_frames_from_video(video_path, frame_queue):
#     """
#     从视频文件捕获帧并将其放入帧队列中。
#
#     参数：
#     video_path -- 视频文件路径
#     frame_queue -- 帧队列，用于存储捕获的帧
#     """
#     cap = cv2.VideoCapture(video_path)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_queue.put(frame)
#         # 添加调试信息
#         print("Captured a frame")
#
#     cap.release()
#     print("Finished capturing frames")
#
#
# def detect_LED_blinking_from_video(frame_queue, roi_size):
#     """
#     从帧队列中检测LED闪烁。
#
#     参数：
#     frame_queue -- 帧队列，用于存储捕获的帧
#     roi_size -- ROI大小，用于检测LED闪烁
#     """
#     roi_status = {}
#     start_time = time.time()
#     end_time = None
#
#     while True:
#         if frame_queue.empty():
#             time.sleep(0.1)
#             continue
#
#         frame = frame_queue.get()
#         print("Processing a frame")  # 调试信息
#
#         # 确认分析函数输出
#         try:
#             _, analysis_results = analyze_image_with_rois(frame, roi_size=roi_size)
#             print(f"Analysis results: {analysis_results}")  # 调试信息
#         except Exception as e:
#             print(f"Error analyzing frame: {e}")
#             continue
#
#         current_time = time.time()
#         on_rois = []
#         unlit_rois = []
#         for result in analysis_results:
#             roi_id = result['roi_id']
#             mean_brightness = result['mean_brightness']
#
#             if mean_brightness > 50:
#                 on_rois.append(roi_id)
#                 if roi_id not in roi_status:
#                     roi_status[roi_id] = 'on'
#                     if all([roi_status.get(i, 'off') == 'on' for i in range(1, roi_id)]):
#                         end_time = time.time()
#                     else:
#                         print(f"Error in lighting sequence at ROI {roi_id}")
#             else:
#                 unlit_rois.append(roi_id)
#                 if roi_id in roi_status and roi_status[roi_id] == 'on':
#                     print(f"Error: ROI {roi_id} turned off")
#
#         all_on = all(status == 'on' for status in roi_status.values())
#         if all_on:
#             end_time = time.time()
#
#         if end_time is not None:
#             total_time = end_time - start_time
#         else:
#             total_time = current_time - start_time
#
#         unlit_rois = [i for i in range(1, max(roi_status.keys()) + 1) if roi_status.get(i, 'off') == 'off']
#
#         # 在图像上显示结果
#         # 定义每行文本的高度
#         line_height = 30
#
#         # 计算每个文本块的行数
#         lighted_text_lines = f"Lighted ROI indexes:\n{on_rois}".split('\n')
#         unlighted_text_lines = f"Unlighted ROI indexes:\n{unlit_rois}".split('\n')
#         total_time_text_lines = f"Total time:\n{total_time:.2f} seconds".split('\n')
#
#         # 绘制点亮的ROI索引
#         y_start = 30
#         for idx, line in enumerate(lighted_text_lines):
#             cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#         # 绘制未点亮的ROI索引，调整起始位置以避免重叠
#         y_start += len(lighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
#         for idx, line in enumerate(unlighted_text_lines):
#             cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#         # 绘制总时间，调整起始位置以避免重叠
#         y_start += len(unlighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
#         for idx, line in enumerate(total_time_text_lines):
#             cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#         # 在终端上显示结果
#         print(f"Lighted ROIs: {on_rois}")
#         print(f"Unlit ROIs: {unlit_rois}")
#         print(f"Total time: {total_time:.2f} seconds")
#
#         cv2.imshow("Video", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#
#
# def main():
#     video_path = "E:\\workspace\\Data\\LED_data\\task2\\Flow_mode1.avi"  # 替换为您的视频文件路径
#
#     # 创建帧队列
#     frame_queue = queue.Queue(maxsize=10000)
#
#     # 启动捕获和检测线程
#     capture_thread = threading.Thread(target=capture_frames_from_video, args=(video_path, frame_queue))
#     detect_thread = threading.Thread(target=detect_LED_blinking_from_video, args=(frame_queue, 20))
#
#     capture_thread.start()
#     detect_thread.start()
#
#     capture_thread.join()
#     detect_thread.join()
#
#
# if __name__ == '__main__':
#     main()

############
import cv2
import numpy as np
import threading
import queue
import time
from collections import namedtuple
import gxipy as gx
from common.Common import object_extraction, object_curve_fitting, curve_division, draw_rectangle_roi_base_on_points
from common.Common import analyze_image_with_rois

def capture_frames_from_video(video_path, frame_queue):
    """
    从视频文件捕获帧并将其放入帧队列中。
    参数：
    video_path -- 视频文件路径
    frame_queue -- 帧队列，用于存储捕获的帧
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()

def detect_LED_blinking_from_video(frame_queue, roi_size):
    """
    从帧队列中检测LED闪烁。
    参数：
    frame_queue -- 帧队列，用于存储捕获的帧
    roi_size -- ROI大小，用于检测LED闪烁
    """
    roi_status = {}
    start_time = time.time()
    end_time = None
    frame_count = 0
    line_height = 30
    lighting_order = []
    max_roi_id = 0

    while True:
        if frame_queue.empty():
            time.sleep(0.1)
            continue

        frame = frame_queue.get()
        frame_count += 1

        try:
            _, analysis_results = analyze_image_with_rois(frame, roi_size=roi_size)
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            continue

        current_time = time.time()
        new_lights = []
        for result in analysis_results:
            roi_id = result['roi_id']
            mean_brightness = result['mean_brightness']

            if mean_brightness > 50:
                if roi_id not in roi_status:
                    roi_status[roi_id] = 'on'
                    new_lights.append(roi_id)
                    lighting_order.append(roi_id)
                    if roi_id > max_roi_id:
                        max_roi_id = roi_id
            else:
                if roi_id in roi_status and roi_status[roi_id] == 'on':
                    print(f"Error: ROI {roi_id} turned off")

        if new_lights:
            end_time = None
        else:
            if end_time is None:
                end_time = current_time

        if end_time is not None:
            total_time = end_time - start_time
        else:
            total_time = current_time - start_time

        unlit_rois = [i for i in range(1, max_roi_id + 1) if roi_status.get(i, 'off') == 'off']

        # 在图像上显示结果
        lighted_text = f"Lighted ROI indexes: {lighting_order}"
        unlighted_text = f"Unlighted ROI indexes: {unlit_rois}"
        total_time_text = f"Total time: {total_time:.2f} seconds"

        y_start = 30
        for text, color in [(lighted_text, (0, 255, 0)), (unlighted_text, (0, 0, 255)), (total_time_text, (255, 255, 255))]:
            for line in text.split('\n'):
                cv2.putText(frame, line, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_start += line_height + 10

        print(f"Lighted ROIs: {lighting_order}")
        print(f"Unlit ROIs: {unlit_rois}")
        print(f"Total time: {total_time:.2f} seconds")

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    video_path = "E:\\workspace\\Data\\LED_data\\task2\\Flow_mode1"  # 替换为您的视频文件路径

    # 创建帧队列
    frame_queue = queue.Queue(maxsize=100)

    # 启动捕获和检测线程
    capture_thread = threading.Thread(target=capture_frames_from_video, args=(video_path, frame_queue))
    detect_thread = threading.Thread(target=detect_LED_blinking_from_video, args=(frame_queue, 20))

    capture_thread.start()
    detect_thread.start()

    capture_thread.join()
    detect_thread.join()

if __name__ == '__main__':
    main()


