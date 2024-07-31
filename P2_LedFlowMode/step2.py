import queue
import threading
from V1_Video2Queue import video2Queue
from F4_FramesQueueProcess import F4FramesQueueProcess




def step2(VideoPath):
    # 创建一个线程安全的队列
    frame_queue = queue.Queue(maxsize=100)

    # 创建停止事件
    stop_event = threading.Event()

    # 启动视频读取线程
    reader_thread = threading.Thread(target=video2Queue, args=(video_path, frame_queue, stop_event))
    reader_thread.start()

    # 启动帧处理线程
    processor_thread = threading.Thread(target=F4FramesQueueProcess, args=(frame_queue, stop_event))
    processor_thread.start()

    # 等待线程结束
    reader_thread.join()
    stop_event.set()
    processor_thread.join()


if __name__ == "__main__":
    video_path = r"E:\workspace\Data\LED_data\task2\6.mp4"
    step2(video_path)
