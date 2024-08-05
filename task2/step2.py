import queue
import threading
from V1_Video2Queue import video2Queue
from F4_FramesQueueProcess import F4FramesQueueProcess


def step2(video_path):
    # 创建一个线程安全的队列
    frame_queue = queue.Queue(maxsize=100)

    # 创建停止事件
    stop_event = threading.Event()

    # 启动视频读取线程
    reader_thread = threading.Thread(target=video2Queue, args=(video_path, frame_queue, stop_event))
    reader_thread.start()

    # 创建一个队列来存储处理结果
    results_queue = queue.Queue()

    # 启动队列处理线程
    processor_thread = threading.Thread(target=F4FramesQueueProcess, args=(frame_queue, stop_event, results_queue))
    processor_thread.start()

    # 等待线程结束
    reader_thread.join()
    stop_event.set()
    processor_thread.join()

    # 获取处理结果
    results = results_queue.get()
    return results


if __name__ == "__main__":
    video_path = r"../../../projectData/LED_data/task2/6.mp4"
    step2(video_path)
