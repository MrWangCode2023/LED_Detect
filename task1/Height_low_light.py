#
# ################################################################################################################
# import cv2
# import os
# import numpy as np
# from collections import namedtuple
# ################################################################################################################
# def show_LED_color(color=(0, 0, 0)):
#     # 定义图像的宽度和高度
#     width, height = 640, 640
#     # 创建一个纯色图像，大小为 width x height，数据类型为 uint8
#     color_image = np.full((height, width, 3), color, dtype=np.uint8)
#     return color_image
#
# def object_color_extraction(image):
#     ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
#     filtered_contours, binary, ROI_count = object_extraction(image)
#     roi_color_image = cv2.bitwise_and(image, image, mask=binary)
#     # 计算掩码图像中非零像素的数量
#     nonzero_pixel_count = float(np.count_nonzero(binary))
#
#     # 通道拆分
#     blue_channel = roi_color_image[:, :, 0]
#     green_channel = roi_color_image[:, :, 1]
#     red_channel = roi_color_image[:, :, 2]
#
#     blue_sum = np.sum(blue_channel)
#     green_sum = np.sum(green_channel)
#     red_sum = np.sum(red_channel)
#
#     # 三个通道区域的像素分别求均值
#     ROI_BGR_mean = ()  # 空元组
#     ROI_BGR_mean += (blue_sum / nonzero_pixel_count,)
#     ROI_BGR_mean += (green_sum / nonzero_pixel_count,)
#     ROI_BGR_mean += (red_sum / nonzero_pixel_count,)
#
#     # BGR均值转换为HSV格式
#     bgr_mean = np.array([[ROI_BGR_mean]], dtype=np.uint8)
#     ROI_HSV_mean = cv2.cvtColor(bgr_mean, cv2.COLOR_BGR2HSV)[0][0]
#     brightness = ROI_HSV_mean[2]
#
#     return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count)
#
# def object_extraction(image):
#     ROIResult = namedtuple('ROIResult', ['filtered_contours', 'binary_image', 'roi_count'])
#     # 转换为灰度图像
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 应用高斯模糊
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # 应用二值化
#     _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
#     # 找到轮廓
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     # 初始化掩码
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     # 过滤轮廓，假设我们只保留面积大于2000的轮廓
#     filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 2000]
#     # 在掩码上绘制过滤后的轮廓
#     for contour in filtered_contours:
#         cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#     # 二值图像现在是具有过滤轮廓的掩码
#     binary_image = mask
#     # 计算ROI的数量
#     roi_count = len(filtered_contours)
#
#     # 返回包含所有结果的 namedtuple
#     return ROIResult(filtered_contours, binary_image, roi_count)
#
# ############################## 高亮低亮模式检测 ##############################################
# def height2low_detect(video_path, low_brightness_threshold=-10, hight_brightness_threshold=10):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return
#
#     frame_count = 0
#     previous_brightness = 0
#     high_brightness_count, low_brightness_count = 0, 0
#     h_nums, l_nums = 0, 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         result = object_color_extraction(frame)
#         brightness = result.brightness
#         frame_count += 1
#         brightness_difference = 0
#
#         if frame_count > 1:
#             brightness_difference = brightness - previous_brightness
#             print(f"亮度变化值:{brightness_difference}")
#
#             if brightness_difference > hight_brightness_threshold:
#                 high_brightness_count += brightness
#                 h_nums += 1
#                 print("高亮模式")
#                 text_color = (0, 255, 0)  # 绿色表示高亮模式
#                 mode_text = "Hight brightness mode"
#             elif low_brightness_threshold <= brightness_difference <= hight_brightness_threshold:
#                 print("常亮模式")
#                 text_color = (255, 0, 0)  # 蓝色表示常亮模式
#                 mode_text = "Normal brightness mode"
#             else:
#                 low_brightness_count += brightness
#                 l_nums += 1
#                 print("低亮模式")
#                 text_color = (0, 0, 255)  # 红色表示低亮模式
#                 mode_text = "Low brightness mode"
#
#             cv2.putText(frame, f"Brightness: {brightness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
#             cv2.putText(frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
#         previous_brightness = brightness
#         cv2.imshow('Video', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#     if h_nums:
#         average_high_brightness = high_brightness_count / h_nums
#         print("高亮亮度均值：", average_high_brightness)
#     if l_nums:
#         average_low_brightness = low_brightness_count / l_nums
#         print("低亮亮度均值：", average_low_brightness)
#
#
# if __name__ == "__main__":
#     video_path = "../.../Data/LED_data/task1/task1_1mp4.avi"  # 请替换为实际存在的视频文件路径
#     # E:\workspace\Data\LED_data\task1
#     height2low_detect(video_path)
#
#####################################
import cv2
import os
import numpy as np
from collections import namedtuple

def show_LED_color(color=(0, 0, 0)):
    width, height = 640, 640
    color_image = np.full((height, width, 3), color, dtype=np.uint8)
    return color_image

def object_color_extraction(image):
    ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
    filtered_contours, binary, ROI_count = object_extraction(image)
    roi_color_image = cv2.bitwise_and(image, image, mask=binary)
    nonzero_pixel_count = float(np.count_nonzero(binary))

    blue_channel = roi_color_image[:, :, 0]
    green_channel = roi_color_image[:, :, 1]
    red_channel = roi_color_image[:, :, 2]

    blue_sum = np.sum(blue_channel)
    green_sum = np.sum(green_channel)
    red_sum = np.sum(red_channel)

    ROI_BGR_mean = (
        blue_sum / nonzero_pixel_count,
        green_sum / nonzero_pixel_count,
        red_sum / nonzero_pixel_count
    )

    bgr_mean = np.array([[ROI_BGR_mean]], dtype=np.uint8)
    ROI_HSV_mean = cv2.cvtColor(bgr_mean, cv2.COLOR_BGR2HSV)[0][0]
    brightness = ROI_HSV_mean[2]

    return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count)

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

def height2low_detect(video_path, low_brightness_threshold=-10, high_brightness_threshold=10):
    if not os.path.isfile(video_path):
        print("视频文件不存在:", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件:", video_path)
        return

    frame_count = 0
    previous_brightness = 0
    high_brightness_count, low_brightness_count = 0, 0
    h_nums, l_nums = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = object_color_extraction(frame)
        brightness = result.brightness
        frame_count += 1
        brightness_difference = 0

        if frame_count > 1:
            brightness_difference = brightness - previous_brightness
            print(f"亮度变化值: {brightness_difference}")

            if brightness_difference > high_brightness_threshold:
                high_brightness_count += brightness
                h_nums += 1
                print("高亮模式")
                text_color = (0, 255, 0)
                mode_text = "High brightness mode"
            elif low_brightness_threshold <= brightness_difference <= high_brightness_threshold:
                print("常亮模式")
                text_color = (255, 0, 0)
                mode_text = "Normal brightness mode"
            else:
                low_brightness_count += brightness
                l_nums += 1
                print("低亮模式")
                text_color = (0, 0, 255)
                mode_text = "Low brightness mode"

            cv2.putText(frame, f"Brightness: {brightness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        previous_brightness = brightness
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if h_nums:
        average_high_brightness = high_brightness_count / h_nums
        print("高亮亮度均值：", average_high_brightness)
    if l_nums:
        average_low_brightness = low_brightness_count / l_nums
        print("低亮亮度均值：", average_low_brightness)

if __name__ == "__main__":
    video_path = r"E:\workspace\Data\LED_data\task1\1.avi"  # 请替换为实际存在的视频文件的绝对路径
    height2low_detect(video_path)
