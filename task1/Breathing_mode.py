# import cv2
# import numpy as np
# import gxipy as gx
# import queue
# import threading
# import time
# from collections import namedtuple
#
# def object_extraction(image):
#     ROIResult = namedtuple('ROIResult', ['filtered_contours', 'binary_image', 'roi_count'])
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 2000]
#     for contour in filtered_contours:
#         cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#     binary_image = mask
#     roi_count = len(filtered_contours)
#     return ROIResult(filtered_contours, binary_image, roi_count)
#
# def object_color_extraction(image):
#     ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
#     filtered_contours, binary, ROI_count = object_extraction(image)
#     roi_color_image = cv2.bitwise_and(image, image, mask=binary)
#     nonzero_pixel_count = float(np.count_nonzero(binary))
#     if nonzero_pixel_count == 0:
#         return ROI((0, 0, 0), (0, 0, 0), 0, ROI_count)
#
#     blue_channel = roi_color_image[:, :, 0]
#     green_channel = roi_color_image[:, :, 1]
#     red_channel = roi_color_image[:, :, 2]
#
#     blue_sum = np.sum(blue_channel)
#     green_sum = np.sum(green_channel)
#     red_sum = np.sum(red_channel)
#
#     ROI_BGR_mean = (blue_sum / nonzero_pixel_count, green_sum / nonzero_pixel_count, red_sum / nonzero_pixel_count)
#
#     bgr_mean = np.array([[ROI_BGR_mean]], dtype=np.uint8)
#     ROI_HSV_mean = cv2.cvtColor(bgr_mean, cv2.COLOR_BGR2HSV)[0][0]
#     brightness = ROI_HSV_mean[2]
#
#     return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count)
#
# def find_local_extrema(data):
#     peaks_max = np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1
#     peaks_min = np.where((data[1:-1] < data[:-2]) & (data[1:-1] < data[2:]))[0] + 1
#     return peaks_max, peaks_min
#
# def calculate_periods(peaks):
#     periods = np.diff(peaks)
#     return periods
#
# def LED_breathing_mode_detect(frame_queue):
#     brightness_values = []
#     bgr_means = []
#     period_data = []
#
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             current_roi = object_color_extraction(frame)
#             brightness = current_roi.brightness
#             brightness_values.append(brightness)
#             bgr_means.append(current_roi.ROI_BGR_mean)
#
#             if len(brightness_values) > 10:
#                 smoothed_values = cv2.boxFilter(np.array(brightness_values), -1, (5,5))
#                 peaks_max, peaks_min = find_local_extrema(smoothed_values)
#                 peaks = np.sort(np.concatenate((peaks_max, peaks_min)))
#                 periods = calculate_periods(peaks)
#                 frequencies = 1 / periods if len(periods) > 0 else [0]
#
#                 results = []
#                 for i in range(len(peaks) - 1):
#                     start, end = peaks[i], peaks[i + 1]
#                     period = periods[i]
#                     frequency = frequencies[i]
#                     bgr_mean_period = np.mean(bgr_means[start:end + 1], axis=0)
#                     period_data.append((start, end, period, frequency, bgr_mean_period))
#                     results.append((period, frequency, bgr_mean_period))
#
#                 if len(results) > 0:
#                     period, frequency, bgr_mean = results[-1]
#                     highest_brightness = max(brightness_values)
#                     lowest_brightness = min(brightness_values)
#                     avg_brightness = np.mean(brightness_values)
#
#                     text_lines = [
#                         f"Period: {period:.2f} frames",
#                         f"Frequency: {frequency:.2f} Hz",
#                         f"Highest Brightness: {highest_brightness:.2f}",
#                         f"Lowest Brightness: {lowest_brightness:.2f}",
#                         f"Avg Brightness: {avg_brightness:.2f}",
#                         f"Avg Frequency: {np.mean(frequencies):.2f} Hz"
#                     ]
#
#                     y_offset = 50
#                     for line in text_lines:
#                         cv2.putText(frame, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                         y_offset += 30
#
#                 cv2.imshow("LED Detection", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#
#     # 计算并打印总周期数、平均周期大小、平均频率和平均亮度
#     total_periods = len(period_data)
#     avg_period_size = np.mean([period for _, _, period, _, _ in period_data]) if total_periods > 0 else 0
#     avg_frequency = np.mean([frequency for _, _, _, frequency, _ in period_data]) if total_periods > 0 else 0
#     avg_brightness = np.mean(brightness_values) if len(brightness_values) > 0 else 0
#
#     print(f"总周期个数: {total_periods}")
#     print(f"平均周期大小: {avg_period_size:.2f} frames")
#     print(f"平均频率: {avg_frequency:.2f} Hz")
#     print(f"平均亮度: {avg_brightness:.2f}")
#
#     cv2.destroyAllWindows()
#
# def capture_frames(cam, frame_queue):
#     while True:
#         raw_image = cam.data_stream[0].get_image()
#
#         if raw_image is None or raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
#             continue
#
#         numpy_image = raw_image.get_numpy_array()
#         if numpy_image is None:
#             continue
#
#         pixel_format = raw_image.get_pixel_format()
#         if pixel_format == gx.GxPixelFormatEntry.RGB8:
#             frame = numpy_image
#         elif pixel_format == gx.GxPixelFormatEntry.BAYER_RG8:
#             frame = cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_RG2RGB)
#         else:
#             continue
#
#         frame_queue.put(frame)
#
# def main():
#     device_manager = gx.DeviceManager()
#     dev_num, dev_info_list = device_manager.update_device_list()
#
#     if dev_num == 0:
#         print("没有检测到设备")
#         return
#
#     cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))
#     cam.TriggerMode
#     cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
#     cam.stream_on()
#
#     frame_queue = queue.Queue(maxsize=100)
#     capture_thread = threading.Thread(target=capture_frames, args=(cam, frame_queue))
#     detect_thread = threading.Thread(target=LED_breathing_mode_detect, args=(frame_queue,))
#
#     capture_thread.start()
#     detect_thread.start()
#
#     capture_thread.join()
#     detect_thread.join()
#
#     cam.stream_off()
#     cam.close_device()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     main()

# import cv2
# import numpy as np
# import gxipy as gx
# import queue
# import threading
# import time
# from collections import namedtuple
#
# def object_extraction(image):
#     ROIResult = namedtuple('ROIResult', ['filtered_contours', 'binary_image', 'roi_count'])
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 2000]
#     for contour in filtered_contours:
#         cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#     binary_image = mask
#     roi_count = len(filtered_contours)
#     return ROIResult(filtered_contours, binary_image, roi_count)
#
# def object_color_extraction(image):
#     ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count'])
#     filtered_contours, binary, ROI_count = object_extraction(image)
#     roi_color_image = cv2.bitwise_and(image, image, mask=binary)
#     nonzero_pixel_count = float(np.count_nonzero(binary))
#     if nonzero_pixel_count == 0:
#         return ROI((0, 0, 0), (0, 0, 0), 0, ROI_count)
#
#     blue_channel = roi_color_image[:, :, 0]
#     green_channel = roi_color_image[:, :, 1]
#     red_channel = roi_color_image[:, :, 2]
#
#     blue_sum = np.sum(blue_channel)
#     green_sum = np.sum(green_channel)
#     red_sum = np.sum(red_channel)
#
#     ROI_BGR_mean = (blue_sum / nonzero_pixel_count, green_sum / nonzero_pixel_count, red_sum / nonzero_pixel_count)
#
#     bgr_mean = np.array([[ROI_BGR_mean]], dtype=np.uint8)
#     ROI_HSV_mean = cv2.cvtColor(bgr_mean, cv2.COLOR_BGR2HSV)[0][0]
#     brightness = ROI_HSV_mean[2]
#
#     return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count)
#
# def find_local_extrema(data):
#     peaks_max = np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1
#     peaks_min = np.where((data[1:-1] < data[:-2]) & (data[1:-1] < data[2:]))[0] + 1
#     return peaks_max, peaks_min
#
# def calculate_periods(peaks):
#     periods = np.diff(peaks)
#     return periods
#
# def LED_breathing_mode_detect(frame_queue):
#     brightness_values = []
#     bgr_means = []
#     period_data = []
#
#     last_bright_rois = set()
#     start_time = time.time()
#     end_time = None
#
#     while True:
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             current_roi = object_color_extraction(frame)
#             brightness = current_roi.brightness
#             brightness_values.append(brightness)
#             bgr_means.append(current_roi.ROI_BGR_mean)
#
#             new_bright_rois = {idx + 1 for idx, contour in enumerate(current_roi.filtered_contours)
#                                if cv2.contourArea(contour) > 2000}
#
#             newly_turned_on = new_bright_rois - last_bright_rois
#             never_turned_on = set(range(1, current_roi.ROI_count + 1)) - new_bright_rois
#
#             if len(newly_turned_on) > 0:
#                 end_time = time.time()
#
#             print(f"Newly turned on ROIs: {sorted(newly_turned_on)}")
#             print(f"Never turned on ROIs: {sorted(never_turned_on)}")
#
#             last_bright_rois = new_bright_rois
#
#             if len(brightness_values) > 10:
#                 smoothed_values = cv2.boxFilter(np.array(brightness_values), -1, (5, 5))
#                 peaks_max, peaks_min = find_local_extrema(smoothed_values)
#                 peaks = np.sort(np.concatenate((peaks_max, peaks_min)))
#                 periods = calculate_periods(peaks)
#                 frequencies = 1 / periods if len(periods) > 0 else [0]
#
#                 results = []
#                 for i in range(len(peaks) - 1):
#                     start, end = peaks[i], peaks[i + 1]
#                     period = periods[i]
#                     frequency = frequencies[i]
#                     bgr_mean_period = np.mean(bgr_means[start:end + 1], axis=0)
#                     period_data.append((start, end, period, frequency, bgr_mean_period))
#                     results.append((period, frequency, bgr_mean_period))
#
#                 if len(results) > 0:
#                     period, frequency, bgr_mean = results[-1]
#                     highest_brightness = max(brightness_values)
#                     lowest_brightness = min(brightness_values)
#                     avg_brightness = np.mean(brightness_values)
#
#                     text_lines = [
#                         f"Period: {period:.2f} frames",
#                         f"Frequency: {frequency:.2f} Hz",
#                         f"Highest Brightness: {highest_brightness:.2f}",
#                         f"Lowest Brightness: {lowest_brightness:.2f}",
#                         f"Avg Brightness: {avg_brightness:.2f}",
#                         f"Avg Frequency: {np.mean(frequencies):.2f} Hz"
#                     ]
#
#                     y_offset = 50
#                     for line in text_lines:
#                         cv2.putText(frame, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                         y_offset += 30
#
#                 cv2.imshow("LED Detection", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#
#     # 计算并打印总周期数、平均周期大小、平均频率和平均亮度
#     total_periods = len(period_data)
#     avg_period_size = np.mean([period for _, _, period, _, _ in period_data]) if total_periods > 0 else 0
#     avg_frequency = np.mean([frequency for _, _, _, frequency, _ in period_data]) if total_periods > 0 else 0
#     avg_brightness = np.mean(brightness_values) if len(brightness_values) > 0 else 0
#
#     total_time = end_time - start_time if end_time is not None else time.time() - start_time
#
#     print(f"总周期个数: {total_periods}")
#     print(f"平均周期大小: {avg_period_size:.2f} frames")
#     print(f"平均频率: {avg_frequency:.2f} Hz")
#     print(f"平均亮度: {avg_brightness:.2f}")
#     print(f"亮灯总时间: {total_time:.2f} seconds")
#
#     cv2.destroyAllWindows()
#
# def capture_frames(cam, frame_queue):
#     while True:
#         raw_image = cam.data_stream[0].get_image()
#
#         if raw_image is None or raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
#             continue
#
#         numpy_image = raw_image.get_numpy_array()
#         if numpy_image is None:
#             continue
#
#         pixel_format = raw_image.get_pixel_format()
#         if pixel_format == gx.GxPixelFormatEntry.RGB8:
#             frame = numpy_image
#         elif pixel_format == gx.GxPixelFormatEntry.BAYER_RG8:
#             frame = cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_RG2RGB)
#         else:
#             continue
#
#         frame_queue.put(frame)
#
# def main():
#     device_manager = gx.DeviceManager()
#     dev_num, dev_info_list = device_manager.update_device_list()
#
#     if dev_num == 0:
#         print("没有检测到设备")
#         return
#
#     cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))
#     cam.TriggerMode
#     cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
#     cam.stream_on()
#
#     frame_queue = queue.Queue(maxsize=100)
#     capture_thread = threading.Thread(target=capture_frames, args=(cam, frame_queue))
#     detect_thread = threading.Thread(target=LED_breathing_mode_detect, args=(frame_queue,))
#
#     capture_thread.start()
#     detect_thread.start()
#
#     capture_thread.join()
#     detect_thread.join()
#
#     cam.stream_off()
#     cam.close_device()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     main()

import cv2
import numpy as np
import gxipy as gx
import queue
import threading
import time
from collections import namedtuple
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

def object_extraction(image):
    ROIResult = namedtuple('ROIResult', ['filtered_contours', 'binary_image', 'roi_count'])
    img = np.copy(image)
    # 设置边框参数
    border_color = (0, 0, 0)  # 边框颜色，这里为绿色
    border_thickness = 12  # 边框厚度，单位为像素
    # 计算边框的位置和大小
    height, width, _ = img.shape
    border_top = border_left = 0
    border_bottom = height - 1
    border_right = width - 1
    # 绘制边框
    cv2.rectangle(img, (border_left, border_top), (border_right, border_bottom), border_color,
                  border_thickness)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # _, binary = cv.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(blurred, threshold1=60, threshold2=180)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= 200:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        binary = mask
    roi_count = len(filtered_contours)

    # cv.imshow("gray image", gray)
    # cv.imshow("binary", binary)
    # cv.imshow("edge", edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return ROIResult(filtered_contours, binary, roi_count)

    # return filtered_contours, binary, roi_count

# def object_extraction(image):
#     ROIResult = namedtuple('ROIResult', ['filtered_contours', 'binary_image', 'roi_count'])
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 2000]
#     for contour in filtered_contours:
#         cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#     binary_image = mask
#     roi_count = len(filtered_contours)
#     return ROIResult(filtered_contours, binary_image, roi_count)

def object_color_extraction(image):
    ROI = namedtuple('ROI', ['ROI_BGR_mean', 'ROI_HSV_mean', 'brightness', 'ROI_count', 'filtered_contours'])
    filtered_contours, binary, ROI_count = object_extraction(image)
    roi_color_image = cv2.bitwise_and(image, image, mask=binary)
    nonzero_pixel_count = float(np.count_nonzero(binary))
    if nonzero_pixel_count == 0:
        return ROI((0, 0, 0), (0, 0, 0), 0, ROI_count, [])

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

    return ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count, filtered_contours)

def find_local_extrema(data):
    peaks_max = np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1
    peaks_min = np.where((data[1:-1] < data[:-2]) & (data[1:-1] < data[2:]))[0] + 1
    return peaks_max, peaks_min

def calculate_periods(peaks):
    periods = np.diff(peaks)
    return periods

def optimal_gmm(bgr_values, n_jobs=-1):
    """
    Perform GMM clustering on BGR values and automatically determine the optimal number of components.

    Parameters:
    - bgr_values: A numpy array of shape (n_samples, 3) containing BGR pixel values.
    - n_jobs: Number of parallel jobs to run. -1 means using all processors.

    Returns:
    - best_n_components: Optimal number of components.
    - best_labels: Cluster labels for each pixel.
    - best_centers: Cluster centers (BGR values).
    - best_silhouette_score: Best silhouette score achieved.
    """
    best_n_components = 0
    best_labels = None
    best_centers = None
    best_silhouette_score = -1

    # Define range for number of components
    n_components_range = range(2, min(10, len(bgr_values)))  # Ensure n_components <= n_samples

    def fit_gmm(n_components):
        gmm = GaussianMixture(n_components=n_components, max_iter=100, random_state=42)
        gmm.fit(bgr_values)
        labels = gmm.predict(bgr_values)
        centers = gmm.means_
        silhouette_avg = silhouette_score(bgr_values, labels)
        return n_components, labels, centers, silhouette_avg

    results = Parallel(n_jobs=n_jobs)(delayed(fit_gmm)(n_components) for n_components in n_components_range)

    for n_components, labels, centers, silhouette_avg in results:
        # print(f"n_components: {n_components}, silhouette_avg: {silhouette_avg}")
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_components = n_components
            best_labels = labels
            best_centers = np.rint(centers).astype(int)
            num_centers = len(best_centers)

    # return best_n_components, best_silhouette_score, best_labels, best_centers
    return best_centers, num_centers


def LED_breathing_mode_detect(frame_queue):
    brightness_values = []
    bgr_means = []
    period_data = []

    last_bright_rois = set()
    start_time = time.time()
    end_time = None

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # ROI(ROI_BGR_mean, ROI_HSV_mean, brightness, ROI_count, filtered_contours)
            current_roi = object_color_extraction(frame)
            brightness = current_roi.brightness
            brightness_values.append(brightness)
            bgr_means.append(current_roi.ROI_BGR_mean)  # 存储每一帧的目标BGR均值

            new_bright_rois = {idx + 1 for idx, contour in enumerate(current_roi.filtered_contours)
                               if cv2.contourArea(contour) > 2000}

            newly_turned_on = new_bright_rois - last_bright_rois
            never_turned_on = set(range(1, current_roi.ROI_count + 1)) - new_bright_rois

            if len(newly_turned_on) > 0:
                end_time = time.time()

            # print(f"Newly turned on ROIs: {sorted(newly_turned_on)}")
            # print(f"Never turned on ROIs: {sorted(never_turned_on)}")

            last_bright_rois = new_bright_rois

            if len(brightness_values) > 10:
                smoothed_values = cv2.boxFilter(np.array(brightness_values), -1, (5, 5))
                peaks_max, peaks_min = find_local_extrema(smoothed_values)
                peaks = np.sort(np.concatenate((peaks_max, peaks_min)))
                periods = calculate_periods(peaks)
                frequencies = 1 / periods if len(periods) > 0 else [0]

                results = []
                for i in range(len(peaks) - 1):
                    start, end = peaks[i], peaks[i + 1]
                    period = periods[i]
                    frequency = frequencies[i]
                    bgr_mean_period = np.mean(bgr_means[start:end + 1], axis=0)
                    period_data.append((start, end, period, frequency, bgr_mean_period))
                    results.append((period, frequency, bgr_mean_period))

                if len(results) > 0:
                    period, frequency, bgr_mean = results[-1]
                    highest_brightness = max(brightness_values)
                    lowest_brightness = min(brightness_values)
                    avg_brightness = np.mean(brightness_values)

                    text_lines = [
                        f"Period: {period:.2f} frames",
                        f"Frequency: {frequency:.2f} Hz",
                        f"Highest Brightness: {highest_brightness:.2f}",
                        f"Lowest Brightness: {lowest_brightness:.2f}",
                        f"Avg Brightness: {avg_brightness:.2f}",
                        f"Avg Frequency: {np.mean(frequencies):.2f} Hz"
                    ]

                    y_offset = 50
                    for line in text_lines:
                        cv2.putText(frame, line, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 30

                cv2.imshow("LED Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    best_centers, num_centers = optimal_gmm(bgr_means, n_jobs=-1)

    # 计算并打印总周期数、平均周期大小、平均频率和平均亮度
    total_periods = len(period_data)
    avg_period_size = np.mean([period for _, _, period, _, _ in period_data]) if total_periods > 0 else 0
    avg_frequency = np.mean([frequency for _, _, _, frequency, _ in period_data]) if total_periods > 0 else 0
    avg_brightness = np.mean(brightness_values) if len(brightness_values) > 0 else 0

    total_time = end_time - start_time if end_time is not None else time.time() - start_time

    print(f"总周期个数: {total_periods}")
    print(f"平均周期大小: {avg_period_size:.2f} frames")
    print(f"平均频率: {avg_frequency:.2f} Hz")
    print(f"平均亮度: {avg_brightness:.2f}")
    print(f"亮灯总时间: {total_time:.2f} seconds")
    print(f"检测到的颜色：\n{best_centers}")
    print(f"检测到的颜色个数：{num_centers}")

    cv2.destroyAllWindows()

def capture_frames(cam, frame_queue):
    while True:
        raw_image = cam.data_stream[0].get_image()

        if raw_image is None or raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
            continue

        numpy_image = raw_image.get_numpy_array()
        if numpy_image is None:
            continue

        pixel_format = raw_image.get_pixel_format()
        if pixel_format == gx.GxPixelFormatEntry.RGB8:
            frame = numpy_image
        elif pixel_format == gx.GxPixelFormatEntry.BAYER_RG8:
            frame = cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_RG2RGB)
        else:
            continue

        frame_queue.put(frame)

def main():
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()

    if dev_num == 0:
        print("没有检测到设备")
        return

    cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))
    cam.TriggerMode
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    cam.stream_on()

    frame_queue = queue.Queue(maxsize=100)
    capture_thread = threading.Thread(target=capture_frames, args=(cam, frame_queue))
    detect_thread = threading.Thread(target=LED_breathing_mode_detect, args=(frame_queue,))

    capture_thread.start()
    detect_thread.start()

    capture_thread.join()
    detect_thread.join()

    cam.stream_off()
    cam.close_device()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
