import cv2


def video2Queue(video_path, frame_queue, stop_event):
    """
    从视频文件读取帧并将其放入队列。

    Args:
        video_path: 视频文件的路径
        frame_queue: 存储视频帧的队列。
        stop_event: 停止事件，用于停止视频读取线程。

    Returns:

    """
    # 从视频路径获取到视频数据
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧放入队列
        frame_queue.put(frame)

    cap.release()
    print("视频线程结束")