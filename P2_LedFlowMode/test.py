import cv2
import queue
import threading


def video_reader(video_path, frame_queue, stop_event):
    """
    从视频文件读取帧并将其放入队列。

    参数：
    - video_path: 视频文件的路径。
    - frame_queue: 存储视频帧的队列。
    - stop_event: 停止事件，用于停止视频读取线程。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧放入队列
        frame_queue.put(frame)

    cap.release()
    print("视频读取线程结束")


def process_frames(frame_queue, stop_event):
    """
    处理队列中的帧。

    参数：
    - frame_queue: 存储视频帧的队列。
    - stop_event: 停止事件，用于停止帧处理线程。
    """
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
            # 处理帧（在这里你可以进行图像处理）
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            continue

    cv2.destroyAllWindows()
    print("帧处理线程结束")


# 使用示例
video_path = 'path/to/your/video.mp4'
frame_queue = queue.Queue(maxsize=10)  # 创建一个线程安全的队列
stop_event = threading.Event()  # 创建停止事件

# 启动视频读取线程
reader_thread = threading.Thread(target=video_reader, args=(video_path, frame_queue, stop_event))
reader_thread.start()

# 启动帧处理线程
processor_thread = threading.Thread(target=process_frames, args=(frame_queue, stop_event))
processor_thread.start()

# 等待线程结束
reader_thread.join()
stop_event.set()  # 设置停止事件，停止处理线程
processor_thread.join()
