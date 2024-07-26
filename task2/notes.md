### 项目需求：LED流水检测算法：  
场景：LED灯带会从一端开始亮，亮灯的区域逐渐向另一端扩展，前面亮灯的区域不会熄灭。
需求: 检测亮灯的顺序是否有错误(就是前面一段先亮了再亮后面一段)，是否有范围没亮以及从最开始亮灯到最后全部亮的这个总间隔时间。
视频：{亮灯顺序，没有亮灯的ROI编号，}

输出：{亮灯顺序（ROI）编号， 没有量的的区域(ROI编号)， 全部灯亮的总时间（当没有没有新的ROI判断为亮的时候为结束）}

总输出：
整个检测过程中ROI亮灯的顺序：记录每一帧相对于前一帧新增的亮灯ROI编号。
整个检测过程中从来都没有亮过的ROI编号
整个检测过程中的亮灯时间：当没有新的ROI亮灯的时候判断为亮灯结束

单帧显示数据：
当前已经亮灯了的ROI编号
当前未亮灯编号：小于最大亮灯ROI编号的未亮灯ROI编号
当前亮灯时间


设计一个LED流水检测算法来满足检测亮灯顺序、范围以及总间隔时间的需求。下面是详细的算法流程设计：

### 算法流程设计

1. **初始化和设置**：
   - 导入必要的库：`cv2`, `numpy`, `time`, `collections`, `threading`, `queue`等。
   - 定义数据结构用于存储分析结果。

2. **捕获视频帧**：
   - 从视频文件中逐帧读取视频数据，或者从相机实时捕获帧。
   - 将捕获的帧放入队列`frame_queue`中。

3. **分析帧**：
   - 从帧队列中获取帧。
   - 使用图像处理技术检测ROI区域。
   - 提取每个ROI的亮度信息。

4. **亮灯顺序和数据统计**：
   - 记录每一帧新增的亮灯ROI编号。
   - 统计整个检测过程中从来都没有亮过的ROI编号。
   - 记录亮灯的总时间，当没有新的ROI亮灯的时候判断为亮灯结束。

5. **结果展示**：
   - 在处理后的帧上显示检测结果。
   - 输出统计数据到终端或日志文件。

### 详细流程图

#### 1. 导入库和定义数据结构

```python
import cv2
import numpy as np
import time
import threading
import queue
from collections import defaultdict
```

#### 2. 捕获视频帧

```python
def capture_frames_from_video(video_path, frame_queue):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
```

#### 3. 分析帧

```python
def analyze_frame(frame, roi_size):
    # 假设analyze_image_with_rois函数返回ROI的亮度信息
    _, analysis_results = analyze_image_with_rois(frame, roi_size=roi_size)
    return analysis_results
```

#### 4. 亮灯顺序和数据统计

```python
def detect_led_flow(frame_queue, roi_size):
    roi_status = defaultdict(str)  # 存储每个ROI的亮灯状态 ('on' or 'off')
    lighting_order = []  # 记录亮灯顺序
    start_time = time.time()
    end_time = None
    frame_count = 0
    line_height = 30
    max_roi_id = 0

    while True:
        if frame_queue.empty():
            time.sleep(0.1)
            continue

        frame = frame_queue.get()
        frame_count += 1

        try:
            analysis_results = analyze_frame(frame, roi_size)
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            continue

        current_time = time.time()
        new_lights = []
        for result in analysis_results:
            roi_id = result['roi_id']
            brightness = result['mean_brightness']

            if brightness > 50:
                if roi_status[roi_id] != 'on':
                    roi_status[roi_id] = 'on'
                    new_lights.append(roi_id)
                    lighting_order.append(roi_id)
                    if roi_id > max_roi_id:
                        max_roi_id = roi_id

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
```

#### 5. 结果展示

```python
def display_results(frame, lighting_order, unlit_rois, total_time):
    y_start = 30
    line_height = 30
    
    lighted_text = f"Lighted ROI indexes: {lighting_order}"
    unlighted_text = f"Unlighted ROI indexes: {unlit_rois}"
    total_time_text = f"Total time: {total_time:.2f} seconds"
    
    for text, color in [(lighted_text, (0, 255, 0)), (unlighted_text, (0, 0, 255)), (total_time_text, (255, 255, 255))]:
        for line in text.split('\n'):
            cv2.putText(frame, line, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_start += line_height + 10

    print(f"Lighted ROIs: {lighting_order}")
    print(f"Unlit ROIs: {unlit_rois}")
    print(f"Total time: {total_time:.2f} seconds")
```

#### 6. 主函数

```python
def main():
    video_path = "path_to_your_video"  # 替换为您的视频文件路径
    frame_queue = queue.Queue(maxsize=100000)
    
    capture_thread = threading.Thread(target=capture_frames_from_video, args=(video_path, frame_queue))
    detect_thread = threading.Thread(target=detect_led_flow, args=(frame_queue, 20))
    
    capture_thread.start()
    detect_thread.start()
    
    capture_thread.join()
    detect_thread.join()

if __name__ == '__main__':
    main()
```

### 关键点总结

1. **捕获视频帧**：使用OpenCV从视频文件中逐帧读取数据，并将其放入队列中。
2. **分析帧**：使用自定义函数`analyze_image_with_rois`分析每一帧中的ROI，提取亮度信息。
3. **亮灯顺序和数据统计**：
   - 记录每一帧新增的亮灯ROI编号。
   - 统计整个检测过程中从来都没有亮过的ROI编号。
   - 记录亮灯的总时间，当没有新的ROI亮灯的时候判断为亮灯结束。
4. **结果展示**：在处理后的帧上显示检测结果，并输出统计数据。

这个流程能够有效地检测和分析LED流水灯的亮灯顺序、范围以及总间隔时间，满足检测亮灯顺序、范围以及总间隔时间的需求。