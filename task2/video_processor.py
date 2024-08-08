import pika
import numpy as np
import cv2
import queue
import threading
from F3_FrameProcess import F3FrameProcess
from F4_FramesQueueProcess import F4FramesQueueProcess


def main():
    # 创建一个线程安全的队列
    frame_queue = queue.Queue(maxsize=100)

    # 创建停止事件
    stop_event = threading.Event()

    # 连接到 RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # 声明队列
    channel.queue_declare(queue='frame_queue')

    # 启动帧处理线程
    results_queue = queue.Queue()
    processor_thread = threading.Thread(target=F4FramesQueueProcess, args=(frame_queue, stop_event, results_queue))
    processor_thread.start()

    def callback(ch, method, properties, body):
        # 将接收到的帧数据放入队列
        frame_queue.put(body)

    # 设置消费回调
    channel.basic_consume(queue='frame_queue', on_message_callback=callback, auto_ack=True)

    print('Waiting for frames. To exit press CTRL+C')
    try:
        channel.start_consuming()  # 开始消费消息
    except KeyboardInterrupt:
        stop_event.set()  # 设置停止事件
        processor_thread.join()  # 等待处理线程结束
        channel.stop_consuming()  # 停止消费
    finally:
        connection.close()  # 关闭连接

if __name__ == '__main__':
    main()
