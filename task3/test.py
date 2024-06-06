############################### 同质性视频流检测 ####################################
# import cv2
# import numpy as np
# from common.Common import object_curve_fitting
# from common.Common import object_extraction
# from common.Common import curve_division
# from common.Common import draw_rectangle_roi_base_on_points
# import threading
# import queue
# import gxipy as gx
#
# def analyze_image_with_rois(image, num_divisions=50, roi_size=20, brightness_threshold=50):
#     # 调用绘制ROI的函数，获取带有绘制ROI的图像和ROI顶点坐标
#     image_with_roi, rois = draw_rectangle_roi_base_on_points(image, num_divisions, roi_size)
#
#     # 用于存储每个ROI的分析结果
#     analysis_results = []
#
#     # 遍历每个ROI，并为每个ROI分配一个编号
#     for idx, roi in enumerate(rois):
#         # 创建一个与输入图像大小相同的空白掩码
#         mask = np.zeros(image.shape[:2], dtype=np.uint8)
#
#         # 将ROI顶点坐标转换为int32类型
#         roi_corners = np.array(roi, dtype=np.int32)
#
#         # 在掩码上填充ROI多边形区域，将ROI区域设置为白色
#         cv2.fillPoly(mask, [roi_corners], 255)
#
#         # 使用掩码从图像中提取ROI
#         roi_image = cv2.bitwise_and(image, image, mask=mask)
#
#         # 获取ROI区域的所有像素
#         roi_pixels = roi_image[mask == 255]
#
#         # 如果ROI区域没有像素，跳过该ROI
#         if len(roi_pixels) == 0:
#             continue
#
#         # 计算亮度统计
#         # 将ROI图像转换为灰度图像
#         gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
#
#         # 计算灰度图中ROI区域的平均亮度
#         mean_brightness = np.mean(gray[mask == 255])
#
#         # 计算灰度图中ROI区域的最大亮度
#         max_brightness = np.max(gray[mask == 255])
#
#         # 计算灰度图中ROI区域的最小亮度
#         min_brightness = np.min(gray[mask == 255])
#
#         # 计算亮度低于阈值的像素比例
#         low_brightness_ratio = np.sum(gray[mask == 255] < brightness_threshold) / len(roi_pixels)
#
#         # 计算平均颜色
#         # 计算ROI区域的平均颜色（BGR）
#         mean_color = np.mean(roi_pixels, axis=0).astype(int).tolist()
#
#         # 将结果添加到分析结果列表中
#         analysis_results.append({
#             'roi_id': idx + 1,  # 添加ROI编号
#             'mean_brightness': mean_brightness,
#             'max_brightness': max_brightness,
#             'min_brightness': min_brightness,
#             'mean_color': mean_color,
#             'low_brightness_ratio': low_brightness_ratio
#         })
#
#     return image_with_roi, analysis_results  # 返回带有绘制ROI的图像和分析结果
#
# def main():
#     # 初始化相机设备管理器
#     device_manager = gx.DeviceManager()
#
#     # 更新设备列表
#     dev_num, dev_info_list = device_manager.update_device_list()
#
#     # 如果没有检测到设备，打印提示信息并返回
#     if dev_num == 0:
#         print("没有检测到设备")
#         return
#
#     # 打开第一个检测到的设备
#     cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))
#
#     # 设置相机的触发模式为关闭状态
#     cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
#
#     # 打开相机流
#     cam.stream_on()
#
#     # 创建一个队列，用于存储图像帧
#     frame_queue = queue.Queue(maxsize=100)
#
#     def capture_frames():
#         while True:
#             # 从相机的数据流中获取一张图像
#             raw_image = cam.data_stream[0].get_image()
#
#             # 检查图像是否为空或图像状态是否不完整，如果是则继续获取下一张图像
#             if raw_image is None or raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
#                 continue
#
#             # 将获取到的图像转换为NumPy数组
#             numpy_image = raw_image.get_numpy_array()
#             if numpy_image is None:
#                 continue
#
#             # 获取图像的像素格式
#             pixel_format = raw_image.get_pixel_format()
#
#             # 根据图像的像素格式进行相应的处理
#             if pixel_format == gx.GxPixelFormatEntry.RGB8:
#                 frame = numpy_image  # 如果图像格式是RGB8，直接使用该图像
#             elif pixel_format == gx.GxPixelFormatEntry.BAYER_RG8:
#                 # 如果图像格式是BAYER_RG8，将其转换为RGB格式
#                 frame = cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_RG2RGB)
#             else:
#                 continue  # 如果图像格式不是支持的格式，跳过该图像
#
#             # 将处理后的图像帧放入队列中，以便其他线程进行处理
#             frame_queue.put(frame)
#
#     def process_frames():
#         while True:
#             # 从队列中获取一帧图像
#             frame = frame_queue.get()
#
#             # 分析图像中的ROI
#             image_with_roi, analysis_results = analyze_image_with_rois(frame)
#
#             # 显示分析结果
#             for result in analysis_results:
#                 text = (f"ROI {result['roi_id']}: "
#                         f"Mean: {result['mean_brightness']:.2f}, "
#                         f"Max: {result['max_brightness']:.2f}, "
#                         f"Min: {result['min_brightness']:.2f}, "
#                         f"Mean Color: {result['mean_color']}, "
#                         f"Low Brightness Ratio: {result['low_brightness_ratio']:.2f}")
#
#                 # 在图像上绘制结果文字
#                 cv2.putText(image_with_roi, text, (10, 30 + result['roi_id'] * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#             # 显示结果图像
#             cv2.imshow('Image with ROIs', image_with_roi)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         cv2.destroyAllWindows()
#
#     # 创建一个线程用于采集图像帧
#     capture_thread = threading.Thread(target=capture_frames)
#
#     # 创建一个线程用于处理图像帧
#     process_thread = threading.Thread(target=process_frames)
#
#     # 启动图像采集线程
#     capture_thread.start()
#
#     # 启动图像处理线程
#     process_thread.start()
#
#     # 等待图像采集线程结束
#     capture_thread.join()
#
#     # 等待图像处理线程结束
#     process_thread.join()
#
#     # 关闭相机流
#     cam.stream_off()
#
#     # 关闭相机设备
#     cam.close_device()
#
#
# if __name__ == '__main__':
#     main()
