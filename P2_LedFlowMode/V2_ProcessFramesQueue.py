from F_ProcessFrame import FProcessFrame


import cv2
import queue
import time

def processFramesQueue(frame_queue, stop_event):
    """
    处理队列中的帧

    Args:
        frame_queue: 存储视频帧的队列。
        stop_event: 停止事件，用于停止帧处理线程。

    Returns:

    """
    count = 0
    framesDatas = []
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)

            # 帧数记录
            count += 1
            print(f"count:{count}")

            # 记录开始时间
            start_time = time.time()

            # 处理帧
            frameData, frameresult = FProcessFrame(frame)

            # 记录结束时间
            end_time = time.time()
            # 计算处理时间

            framesDatas.append(frameData)

            processing_time = end_time - start_time

            print(f"单帧处理时间: {processing_time:.4f}秒")  # 打印处理时间

            # 显示单帧结果
            cv2.imshow("image", frameresult)

            # 允许每帧显示1毫秒，并且在这里检测键盘输入
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except queue.Empty:
            continue

    cv2.destroyAllWindows()
    print("帧处理线程结束")

