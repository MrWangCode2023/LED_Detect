import cv2
import numpy as np
import time
from common.Common import analyze_image_with_rois

def detect_LED_blinking_from_video(video_path, roi_size):
    """
    从视频文件中捕获帧并检测LED闪烁。

    参数：
    video_path -- 视频文件路径
    roi_size -- ROI的大小
    """
    roi_status = {}
    start_time = time.time()
    end_time = None

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 分析图像中的ROI
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
        y_start += len(lighted_text_lines) * line_height + 10
        for idx, line in enumerate(unlighted_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 绘制总时间，调整起始位置以避免重叠
        y_start += len(unlighted_text_lines) * line_height + 10
        for idx, line in enumerate(total_time_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 在终端上显示结果
        print(f"Lighted ROIs: {on_rois}")
        print(f"Unlit ROIs: {unlit_rois}")
        print(f"Total time: {total_time:.2f} seconds")

        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = "E:\\workspace\\Data\\LED_data\\task1\\1.avi"  # 替换为实际的视频文件路径
    detect_LED_blinking_from_video(video_path, roi_size=20)
