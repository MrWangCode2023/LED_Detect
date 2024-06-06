import cv2
import numpy as np
from collections import namedtuple
import time
import gxipy as gx
import threading
import queue

# 1. LED区域检测
def object_extraction(image):
    ROIResult = namedtuple('ROIResult', ['filtered_contours', 'binary_image', 'roi_count'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 2000]
    for contour in filtered_contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    binary_image = mask
    roi_count = len(filtered_contours)
    return ROIResult(filtered_contours, binary_image, roi_count)

# 2. 获取目标区域像素值
def object_color_extraction(image):
    ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
    filtered_contours, binary, ROI_count = object_extraction(image)
    roi_color_image = cv2.bitwise_and(image, image, mask=binary)
    nonzero_pixel_count = float(np.count_nonzero(binary))
    if nonzero_pixel_count == 0:
        return ROI((0, 0, 0), (0, 0, 0), 0, ROI_count)
    blue_channel = roi_color_image[:, :, 0]
    green_channel = roi_color_image[:, :, 1]
    red_channel = roi_color_image[:, :, 2]
    blue_sum = np.sum(blue_channel)
    green_sum = np.sum(green_channel)
    red_sum = np.sum(red_channel)
    ROI_BGR_mean = (blue_sum / nonzero_pixel_count, green_sum / nonzero_pixel_count, red_sum / nonzero_pixel_count)
    bgr_mean = np.array([[ROI_BGR_mean]], dtype=np.uint8)
    ROI_HSV_mean = cv2.cvtColor(bgr_mean, cv2.COLOR_BGR2HSV)[0][0]
    brightness = ROI_HSV_mean[2]
    return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count)

# 3. 用窗口展示像素值的颜色
def show_LED_color(color=(0, 0, 0)):
    width, height = 640, 640
    color_image = np.full((height, width, 3), color, dtype=np.uint8)
    return color_image

# 4. 目标区域曲线拟合
def object_curve_fitting(image):
    curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
    filtered_contours, binary, area_count = object_extraction(image)
    binary_image = binary.copy()
    curve_image = cv2.ximgproc.thinning(binary_image)
    nonzero_pixels = np.nonzero(curve_image)
    curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
    curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)
    return curve(curve_image, curve_coordinates, curve_length)

# 5. 对图像进行自适应resize
def auto_resize(image, new_shape=(640, 640)):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    k = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_size = int(round(shape[1] * k)), int(round(shape[0] * k))
    if shape[::-1] != new_size:
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return image

# 6. 对曲线进行等分操作
def curve_division(curve_length, curve_coordinates, num_divisions=30):
    divided_points = []
    segment_length = curve_length / num_divisions
    accumulated_length = 0.0
    divided_point = curve_coordinates[0]
    divided_points.append(divided_point)
    coordinates_num = len(curve_coordinates)
    for i in range(coordinates_num-1):
        distance = np.linalg.norm(curve_coordinates[i] - curve_coordinates[i+1])
        accumulated_length += distance
        if accumulated_length >= segment_length:
            divided_points.append(curve_coordinates[i])
            accumulated_length = 0.0
    return divided_points

# 7. LED亮灭模式检测
def detect_LED_blinking_from_camera(frame_queue, light_threshold=10, stable_threshold=3):
    # 初始化亮度列表、亮度为ON的帧列表、闪烁计数器、总处理时间、帧计数器等
    brightness_list = []
    brightness_on = []
    blink_count = 0
    total_processing_time = 0
    frame_count = 0
    stable_frames = 0  # 记录当前状态连续稳定的帧数
    previous_state = None  # 记录前一帧的亮灭状态

    while True:
        if not frame_queue.empty():
            # 从帧队列中取出一帧图像
            frame = frame_queue.get()

            start_time = time.time()

            # 提取ROI区域并计算亮度
            roi = object_color_extraction(frame)
            brightness_list.append(roi.brightness)  # 用列表存储当前帧的数据

            # 保持亮度列表长度不超过10000
            if len(brightness_list) > 10000:
                brightness_list.pop(0)

            # 当有超过一帧的数据时进行处理
            if len(brightness_list) > 1:
                current_state = brightness_list[-1] > light_threshold  # 当前帧的亮灭状态

                if previous_state is None:
                    previous_state = current_state

                # 判断当前帧与前一帧的亮灭状态是否不同
                if current_state != previous_state:
                    stable_frames += 1  # 如果不同，稳定帧计数器加1
                else:
                    stable_frames = 0  # 如果相同，重置稳定帧计数器

                # 当稳定帧数达到阈值时，判断闪烁
                if stable_frames >= stable_threshold:
                    if previous_state and not current_state:
                        blink_count += 1  # 记录一次闪烁
                        print("Blink Count:", blink_count)
                    previous_state = current_state
                    stable_frames = 0

                # 记录亮度为ON的帧的亮度值
                if current_state:
                    brightness_on.append(brightness_list[-1])

            # 计算最高、最低和平均亮度
            highest_brightness = max(brightness_list)
            lowest_brightness = min(brightness_list)
            average_brightness_on = sum(brightness_on) / len(brightness_on) if brightness_on else 0

            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # 处理时间，单位为毫秒
            total_processing_time += processing_time
            frame_count += 1
            avg_processing_time = total_processing_time / frame_count

            # 分行显示文本信息
            text_lines = [
                f"Blink Count: {blink_count}",  # 输出闪烁次数
                f"Highest Brightness: {highest_brightness}",
                f"Lowest Brightness: {lowest_brightness}",
                f"Avg Brightness (ON): {average_brightness_on:.2f}",
                f"Frame Process Time: {processing_time:.2f} ms"
            ]

            y_offset = 50
            for line in text_lines:
                # 在图像上显示文本信息
                cv2.putText(frame, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset += 50  # 每行文本之间的垂直间距

            # 显示处理后的图像帧
            cv2.imshow("LED Detection", frame)

            # 按下'Q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 清空队列
    while not frame_queue.empty():
        frame_queue.get()

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()



def capture_frames(cam, frame_queue):
    while True:
        # 从相机的数据流中获取一张图像
        raw_image = cam.data_stream[0].get_image()

        # 检查图像是否为空或图像状态是否不完整，如果是则继续获取下一张图像
        if raw_image is None or raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
            continue

        # 将获取到的图像转换为NumPy数组
        numpy_image = raw_image.get_numpy_array()
        if numpy_image is None:
            continue

        # 获取图像的像素格式
        pixel_format = raw_image.get_pixel_format()

        # 根据图像的像素格式进行相应的处理
        if pixel_format == gx.GxPixelFormatEntry.RGB8:
            frame = numpy_image  # 如果图像格式是RGB8，直接使用该图像
        elif pixel_format == gx.GxPixelFormatEntry.BAYER_RG8:
            # 如果图像格式是BAYER_RG8，将其转换为RGB格式
            frame = cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_RG2RGB)
        else:
            continue  # 如果图像格式不是支持的格式，跳过该图像

        # 将处理后的图像帧放入队列中，以便其他线程进行处理
        frame_queue.put(frame)



def main():
    # 初始化相机设备管理器
    device_manager = gx.DeviceManager()

    # 更新设备列表
    dev_num, dev_info_list = device_manager.update_device_list()

    # 如果没有检测到设备，打印提示信息并返回
    if dev_num == 0:
        print("没有检测到设备")
        return

    # 打开第一个检测到的设备
    cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))

    # 设置相机的触发模式为关闭状态
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    # 打开相机流
    cam.stream_on()

    # 创建一个队列，用于存储图像帧
    frame_queue = queue.Queue(maxsize=100)

    # 创建一个线程用于采集图像帧
    capture_thread = threading.Thread(target=capture_frames, args=(cam, frame_queue))

    # 创建一个线程用于检测LED亮灭
    detect_thread = threading.Thread(target=detect_LED_blinking_from_camera, args=(frame_queue, 20))

    # 启动图像采集线程
    capture_thread.start()

    # 启动LED亮灭检测线程
    detect_thread.start()

    # 等待图像采集线程结束
    capture_thread.join()

    # 等待LED亮灭检测线程结束
    detect_thread.join()

    # 关闭相机流
    cam.stream_off()

    # 关闭相机设备
    cam.close_device()

    # 销毁所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
