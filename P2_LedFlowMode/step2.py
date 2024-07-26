import queue
import threading
import json
import cv2
import numpy as np

from V1_Video2Queue import video2Queue
from V2_ProcessFramesQueue import processFramesQueue


# class ColorConverter:
#     @staticmethod
#     def rgb2hsv(rgb):
#         """
#         将RGB值转换为HSV值。
#
#         Args:
#             rgb: 一个包含R、G、B值的元组或列表，例如 (R, G, B)。
#
#         Returns:
#             hsv: 转换后的HSV值，格式为 (H, S, V)。
#         """
#         rgb_normalized = np.array(rgb, dtype=np.float32) / 255.0  # 归一化到0-1范围
#         hsv = cv2.cvtColor(np.array([[rgb_normalized]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0][0]
#         return tuple(hsv)
#
#     @staticmethod
#     def rgb2cie1931(rgb):
#         """
#         将RGB值转换为CIE 1931色彩空间。
#
#         Args:
#             rgb: 一个包含R、G、B值的元组或列表，例如 (R, G, B)。
#
#         Returns:
#             cie1931: 转换后的CIE 1931值，格式为 (X, Y, Z)。
#         """
#         # 使用标准的RGB到XYZ转换矩阵
#         rgb_normalized = np.array(rgb, dtype=np.float32) / 255.0  # 归一化到0-1范围
#         # 定义转换矩阵
#         matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
#                            [0.2126729, 0.7151522, 0.0721750],
#                            [0.0193339, 0.1191920, 0.9503041]])
#         xyz = np.dot(matrix, rgb_normalized)
#         return tuple(xyz)
#
#     @staticmethod
#     def xyz2cielab(xyz):
#         """
#         将RGB值转换为CIELAB值。
#
#         Args:
#             xyz: 一个包含R、G、B值的元组或列表，例如 (R, G, B)。
#
#         Returns:
#             lab: 转换后的CIELAB值，格式为 (L, a, b)。
#         """
#
#         # 将XYZ转换为CIELAB
#         x, y, z = xyz
#         x = x / 95.047  # D65标准
#         y = y / 100.000
#         z = z / 108.883
#
#         # CIELAB转换
#         def f(t):
#             if t > 0.008856:
#                 return t ** (1 / 3)
#             else:
#                 return (t * 7.787) + (16 / 116)
#
#         L = max(0, (116 * f(y)) - 16)
#         a = (f(x) - f(y)) * 500
#         b = (f(y) - f(z)) * 200
#         CIELab = (L, a, b)
#
#         return CIELab
#
# def FroiPixelAnalysis(roi_image, number, darkThreshold=50):
#     """
#     计算ROI区域的R、G、B均值
#
#     Args:
#         roi_image: 提取的ROI区域图像 (BGR格式)。
#
#     Returns:
#         mean_rgb: (R_mean, G_mean, B_mean) 的元组。如果没有有效像素，则返回 (0, 0, 0)。
#     """
#     result = [number]
#     result1 = [number]
#
#     # 确保roi_image不是空的
#     if roi_image is None or roi_image.size == 0:
#         rgb, hsv, tag = (0, 0, 0), (0, 0, 0), 0
#         result.extend([rgb, hsv, tag])
#
#         return result
#
#         # 计算非零像素（即 ROI 区域）的数量
#     # 返回二维数组（对前两个维度的位置像素在第三维度上进行或操作）
#     non_zero_mask = np.any(roi_image != 0, axis=2)
#     num_pixels = np.sum(non_zero_mask)
#
#     if num_pixels == 0:
#         rgb, hsv, tag = (0, 0, 0), (0, 0, 0), 0
#         result.extend([rgb, hsv, tag])
#
#         return result  # 如果没有有效像素，返回零均值
#
#     # 分别计算R、G、B通道的总和
#     sum_r = np.sum(roi_image[..., 2][non_zero_mask])  # BGR中R是第三通道
#     sum_g = np.sum(roi_image[..., 1][non_zero_mask])
#     sum_b = np.sum(roi_image[..., 0][non_zero_mask])
#
#     # 计算单通道均值
#     r = sum_r / num_pixels
#     g = sum_g / num_pixels
#     b = sum_b / num_pixels
#
#     # 计算RGB值
#     rgb = (r, g, b)
#     result.append(rgb)
#     # print(f"ROI区域的R均值、G均值、B均值: {result}\n")
#
#     # 计算HSV值
#     converter = ColorConverter()
#     hsv = converter.rgb2hsv(rgb)
#     h, s, v = hsv
#     result.append(hsv)
#     # print(f"ROI区域的H均值、S均值、V均值: {result}\n")
#
#     # 判断当前roi是否为暗
#     tag = 1 if hsv[0] > darkThreshold else 0
#     result.append(tag)
#     result1.extend([r, g, b, h, s, v, tag])
#     result2 = np.array([r, g, b, hsv[0], hsv[1], hsv[2], tag], dtype=np.float32)
#
#     # print(f"result:{result}")
#     # print(f"result1:{result1}")
#     # print(f"result2:{result2}")
#     #result[编号， rgb, hsv, 亮暗标签]
#     return result
#
#
# def t7LoadRoisFromJson(roisPath):
#     """
#     从 JSON 文件加载矩形框坐标。
#
#     参数：
#     - file_path: 文件路径
#
#     返回：
#     - boxes: 矩形框坐标列表
#     """
#     with open(roisPath, 'r') as f:
#         boxes = json.load(f)
#     return boxes
#
# def F1RoiPixelExtract(image, roi):
#     """
#     根据四边形顶点提取 ROI 区域像素。
#
#     参数：
#     - image: 输入图像
#     - roi: 四边形顶点的列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
#
#     返回：
#     - roi_image: 提取的 ROI 区域图像
#     """
#     # 将四边形顶点转换为多边形轮廓（列表中的单个元素是一个点的数组）
#     pts = np.array(roi, np.int32)
#     pts = pts.reshape((-1, 1, 2))  # 重塑为 (n_points, 1, 2) 的形状
#
#     # 创建一个与图像同大小的掩码，并填充多边形区域
#     roiMask = np.zeros(image.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(roiMask, [pts], (255, 255, 255))
#
#     # 使用掩码提取 ROI
#     roi_image = cv2.bitwise_and(image, image, mask=roiMask)
#
#     # 显示结果
#     # cv2.imshow("roi_image", roi_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     return roi_image
#
# def FProcessFrame(frame):
#     roisPath = r"roisPath.json"
#     rois = t7LoadRoisFromJson(roisPath)
#
#     print(f"result[编号， rgb, hsv, 亮暗标签]")
#     # for roi in rois:
#     for i in range(len(rois)):
#         roi = rois[i]
#
#         # 提取单个roi的像素
#         roi_image = F1RoiPixelExtract(frame, roi)
#         # cv2.imshow("roi_pixel_mask", roi_pixel_mask)
#
#         # 计算每个ROI的RGB均值，HSV
#         frameResult = FroiPixelAnalysis(roi_image, i)
#         print(f"{frameResult}")
#
#         # 在帧上绘制roi
#         box = np.array(roi, dtype=np.int32)
#         points_for_polylines = [box]
#         cv2.polylines(frame, points_for_polylines, isClosed=True, color=(0, 255, 0), thickness=1)
#
#
#     # print(f"rois:{len(rois)}")
#
#     # 显示效果
#     # cv2.imshow("image", frame)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     return frameResult, frame
#
# def video2Queue(video_path, frame_queue, stop_event):
#     """
#     从视频文件读取帧并将其放入队列。
#
#     Args:
#         video_path: 视频文件的路径
#         frame_queue: 存储视频帧的队列。
#         stop_event: 停止事件，用于停止视频读取线程。
#
#     Returns:
#
#     """
#     # 从视频路径获取到视频数据
#     cap = cv2.VideoCapture(video_path)
#
#     if not cap.isOpened():
#         print("无法打开视频")
#         return
#
#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 将帧放入队列
#         frame_queue.put(frame)
#
#     cap.release()
#     print("视频线程结束")
#
# def processFramesQueue(frame_queue, stop_event):
#     """
#     处理队列中的帧
#
#     Args:
#         frame_queue: 存储视频帧的队列。
#         stop_event: 停止事件，用于停止帧处理线程。
#
#     Returns:
#
#     """
#     while not stop_event.is_set() or not frame_queue.empty():
#         try:
#             frame = frame_queue.get(timeout=1)
#
#             # 处理帧
#             framedata, frameresult = FProcessFrame(frame)
#
#             # 显示单帧结果
#             cv2.imshow("image", frameresult)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
#
#             # cv2.imshow("Frame", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#         except queue.Empty:
#             continue
#
#     cv2.destroyAllWindows()
#     print("帧处理线程结束")

def step2(VideoPath):
    # 创建一个线程安全的队列
    frame_queue = queue.Queue(maxsize=100)

    # 创建停止事件
    stop_event = threading.Event()

    # 启动视频读取线程
    reader_thread = threading.Thread(target=video2Queue, args=(video_path, frame_queue, stop_event))
    reader_thread.start()

    # 启动帧处理线程
    processor_thread = threading.Thread(target=processFramesQueue, args=(frame_queue, stop_event))
    processor_thread.start()

    # 等待线程结束
    reader_thread.join()
    stop_event.set()
    processor_thread.join()


if __name__ == "__main__":
    video_path = r"E:\workspace\Data\LED_data\task2\6.mp4"
    step2(video_path)
