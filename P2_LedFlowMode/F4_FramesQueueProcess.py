import numpy as np

from F3_FrameProcess import F3FrameProcess


import cv2
import queue
import time

def F4FramesQueueProcess(frame_queue, stop_event):
    """
    处理队列中的帧

    Args:
        frame_queue: 存储视频帧的队列。
        stop_event: 停止事件，用于停止帧处理线程。

    Returns:

    """
    framesDatas = []
    count = 0
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)

            # # 帧数记录
            count += 1
            # print(f"count:{count}")

            # 处理帧
            frameData, resultShow = F3FrameProcess(frame)
            print(f"frameData of {count} frame:\nnumber  tag  r  g  b  h  s  v\n{frameData}")

            # 统计每一帧数据
            framesDatas.append(frameData)

            # 显示单帧结果
            cv2.imshow("image", resultShow)

            # 允许每帧显示1毫秒，并且在这里检测键盘输入
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except queue.Empty:
            continue

    cv2.destroyAllWindows()

    # 将存储的数据转化为数组
    results2 = np.array(framesDatas)
    results = results2.reshape(count, 8, -1)

    print(f"result[编号， rgb, hsv, 亮暗标签]")
    print(f"framesDatas:{results}")
    print("帧处理线程结束")

    return results


if __name__ == "__main__":
    import queue
    import threading
    from V1_Video2Queue import video2Queue
    from F4_FramesQueueProcess import F4FramesQueueProcess

    video_path = r"E:\workspace\Data\LED_data\task2\6.mp4"

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


