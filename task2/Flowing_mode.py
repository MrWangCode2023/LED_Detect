# import cv2
# import numpy as np
# import threading
# import queue
# import time
# from collections import namedtuple
# import gxipy as gx
#
# def capture_frames(cam, frame_queue):
#     while True:
#         raw_image = cam.data_stream[0].get_image()
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
# def object_extraction(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#
#     filtered_contours = []
#     for contour in contours:
#         if cv2.contourArea(contour) > 2000:
#             filtered_contours.append(contour)
#             cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#     binary = mask
#     ROI_count = len(filtered_contours)
#     return filtered_contours, binary, ROI_count
#
# def object_curve_fitting(image):
#     curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
#     filtered_contours, binary, area_count = object_extraction(image)
#     binary_image = binary.copy()
#     curve_image = cv2.ximgproc.thinning(binary_image)
#     nonzero_pixels = np.nonzero(curve_image)
#
#     if len(nonzero_pixels[0]) == 0:
#         return None
#
#     curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
#     curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)
#     return curve(curve_image, curve_coordinates, curve_length)
#
# def curve_division(curve_length, curve_coordinates, num_divisions=50):
#     points_and_angles = []
#     segment_length = curve_length / num_divisions
#     accumulated_length = 0.0
#     coordinates_num = len(curve_coordinates)
#
#     divided_point = curve_coordinates[0]
#     points_and_angles.append((divided_point, 0))
#
#     for i in range(1, coordinates_num):
#         prev_point = curve_coordinates[i - 1]
#         next_point = curve_coordinates[i]
#         distance = np.linalg.norm(next_point - prev_point)
#         accumulated_length += distance
#
#         while accumulated_length >= segment_length:
#             accumulated_length -= segment_length
#             t = 1 - (accumulated_length / distance)
#             divided_point = (1 - t) * prev_point + t * next_point
#             if len(points_and_angles) == 1:
#                 angle = np.arctan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) * 180 / np.pi
#             else:
#                 angle = points_and_angles[-1][1]
#             points_and_angles.append((divided_point, angle))
#
#     for i in range(1, len(points_and_angles) - 1):
#         prev_point = points_and_angles[i - 1][0]
#         next_point = points_and_angles[i + 1][0]
#         tangent_vector = next_point - prev_point
#         normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
#         angle = np.arctan2(normal_vector[1], normal_vector[0]) * 180 / np.pi
#         points_and_angles[i] = (points_and_angles[i][0], angle)
#
#     return points_and_angles
#
# def draw_rectangle_roi_base_on_points(image, num_divisions=50, roi_size=20):
#     curve = object_curve_fitting(image)
#     if curve is None:
#         return image.copy(), []
#
#     points_and_angles = curve_division(curve.curve_length, curve.curve_coordinates, num_divisions)
#     image_with_roi = image.copy()
#     half_size = roi_size // 2
#     rois = []
#
#     for (point, angle) in points_and_angles:
#         center = (int(point[0]), int(point[1]))
#         size = (roi_size, roi_size)
#         rect = (center, size, angle)
#         box = cv2.boxPoints(rect)
#         box = np.intp(box)
#         cv2.drawContours(image_with_roi, [box], 0, (255, 255, 255), 2)
#         rois.append(box.tolist())
#
#     return image_with_roi, rois
#
# def analyze_image_with_rois(image, num_divisions=50, roi_size=20, brightness_threshold=50):
#     image_with_roi, rois = draw_rectangle_roi_base_on_points(image, num_divisions, roi_size)
#     analysis_results = []
#
#     for idx, roi in enumerate(rois):
#         mask = np.zeros(image.shape[:2], dtype=np.uint8)
#         roi_corners = np.array(roi, dtype=np.int32)
#         cv2.fillPoly(mask, [roi_corners], 255)
#         roi_image = cv2.bitwise_and(image, image, mask=mask)
#         roi_pixels = roi_image[mask == 255]
#
#         if len(roi_pixels) == 0:
#             continue
#
#         gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
#         mean_brightness = round(np.mean(gray[mask == 255], dtype=float), 2)
#         max_brightness = np.max(gray[mask == 255])
#         min_brightness = np.min(gray[mask == 255])
#         low_brightness_ratio = round(np.sum(gray[mask == 255] < brightness_threshold) / len(roi_pixels), 2)
#         mean_color = np.mean(roi_pixels, axis=0).astype(int).tolist()
#
#         analysis_results.append({
#             'roi_id': idx + 1,
#             'mean_brightness': mean_brightness,
#             'max_brightness': max_brightness,
#             'min_brightness': min_brightness,
#             'mean_color': mean_color,
#             'low_brightness_ratio': low_brightness_ratio
#         })
#
#     return image_with_roi, analysis_results
#
# def detect_LED_blinking_from_camera(frame_queue, roi_size):
#     roi_status = {}
#     start_time = time.time()
#     end_time = None
#     all_on_time = None  # 用于记录所有灯亮起的时间
#     never_lit_rois = set()
#     first_frame = True
#
#     lighting_sequence = []  # 记录亮灯顺序
#     never_lit = set()  # 从来没有亮过的ROI
#     last_on_rois = set()  # 上一帧亮灯的ROI
#
#     while True:
#         frame = frame_queue.get()
#         _, analysis_results = analyze_image_with_rois(frame, roi_size=roi_size)
#
#         current_time = time.time()
#         on_rois = set()
#         new_on_rois = set()
#         for result in analysis_results:
#             roi_id = result['roi_id']
#             mean_brightness = result['mean_brightness']
#
#             if mean_brightness > 50:
#                 on_rois.add(roi_id)
#                 if roi_id not in roi_status:
#                     roi_status[roi_id] = 'on'
#                     new_on_rois.add(roi_id)
#                     if all([roi_status.get(i, 'off') == 'on' for i in range(1, roi_id)]):
#                         end_time = time.time()
#                     else:
#                         print(f"Error in lighting sequence at ROI {roi_id}")
#             else:
#                 if roi_id in roi_status and roi_status[roi_id] == 'on':
#                     print(f"Error: ROI {roi_id} turned off")
#
#         if first_frame:
#             never_lit = set(range(1, len(analysis_results) + 1))
#             first_frame = False
#
#         never_lit.difference_update(on_rois)
#
#         if new_on_rois:
#             lighting_sequence.append((current_time - start_time, list(new_on_rois)))
#
#         all_on = all(status == 'on' for status in roi_status.values())
#         if all_on and all_on_time is None:
#             all_on_time = current_time
#
#         if end_time is not None:
#             total_time = end_time - start_time
#         else:
#             total_time = current_time - start_time
#
#         unlit_rois = [i for i in range(1, max(roi_status.keys()) + 1) if roi_status.get(i, 'off') == 'off']
#
#         line_height = 30
#
#         lighted_text_lines = f"Lighted ROI indexes:\n{list(on_rois)}".split('\n')
#         unlighted_text_lines = f"Unlighted ROI indexes:\n{unlit_rois}".split('\n')
#         total_time_text_lines = f"Total time:\n{total_time:.2f} seconds".split('\n')
#
#         y_start = 30
#         for idx, line in enumerate(lighted_text_lines):
#             cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#         y_start += len(lighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
#         for idx, line in enumerate(unlighted_text_lines):
#             cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#         y_start += len(unlighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
#         for idx, line in enumerate(total_time_text_lines):
#             cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
#                         2)
#
#         print(f"Lighted ROIs: {list(on_rois)}")
#         print(f"Unlit ROIs: {unlit_rois}")
#         print(f"Total time: {total_time:.2f} seconds")
#         print(f"Lighting sequence: {lighting_sequence}")
#         # print(f"Never lit ROIs: {list(never_lit)}")
#
#         cv2.imshow("Live", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
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
#     cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
#     cam.stream_on()
#
#     frame_queue = queue.Queue(maxsize=100)
#
#     capture_thread = threading.Thread(target=capture_frames, args=(cam, frame_queue))
#     detect_thread = threading.Thread(target=detect_LED_blinking_from_camera, args=(frame_queue, 20))
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
######################
import cv2
import numpy as np
import threading
import queue
import time
from collections import namedtuple
import gxipy as gx


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


def object_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            filtered_contours.append(contour)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    binary = mask
    ROI_count = len(filtered_contours)
    return filtered_contours, binary, ROI_count


def object_curve_fitting(image):
    curve = namedtuple('curve', ['curve_image', 'curve_coordinates', 'curve_length'])
    filtered_contours, binary, area_count = object_extraction(image)
    binary_image = binary.copy()
    curve_image = cv2.ximgproc.thinning(binary_image)
    nonzero_pixels = np.nonzero(curve_image)

    if len(nonzero_pixels[0]) == 0:
        return None

    curve_coordinates = np.column_stack((nonzero_pixels[1], nonzero_pixels[0]))
    curve_length = cv2.arcLength(np.array(curve_coordinates), closed=False)
    return curve(curve_image, curve_coordinates, curve_length)


def curve_division(curve_length, curve_coordinates, num_divisions=50):
    points_and_angles = []
    segment_length = curve_length / num_divisions
    accumulated_length = 0.0
    coordinates_num = len(curve_coordinates)

    divided_point = curve_coordinates[0]
    points_and_angles.append((divided_point, 0))

    for i in range(1, coordinates_num):
        prev_point = curve_coordinates[i - 1]
        next_point = curve_coordinates[i]
        distance = np.linalg.norm(next_point - prev_point)
        accumulated_length += distance

        while accumulated_length >= segment_length:
            accumulated_length -= segment_length
            t = 1 - (accumulated_length / distance)
            divided_point = (1 - t) * prev_point + t * next_point
            if len(points_and_angles) == 1:
                angle = np.arctan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) * 180 / np.pi
            else:
                angle = points_and_angles[-1][1]
            points_and_angles.append((divided_point, angle))

    for i in range(1, len(points_and_angles) - 1):
        prev_point = points_and_angles[i - 1][0]
        next_point = points_and_angles[i + 1][0]
        tangent_vector = next_point - prev_point
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        angle = np.arctan2(normal_vector[1], normal_vector[0]) * 180 / np.pi
        points_and_angles[i] = (points_and_angles[i][0], angle)

    return points_and_angles


def draw_rectangle_roi_base_on_points(image, num_divisions=50, roi_size=20):
    curve = object_curve_fitting(image)
    if curve is None:
        return image.copy(), []

    points_and_angles = curve_division(curve.curve_length, curve.curve_coordinates, num_divisions)
    image_with_roi = image.copy()
    half_size = roi_size // 2
    rois = []

    for (point, angle) in points_and_angles:
        center = (int(point[0]), int(point[1]))
        size = (roi_size, roi_size)
        rect = (center, size, angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(image_with_roi, [box], 0, (255, 255, 255), 2)
        rois.append(box.tolist())

    return image_with_roi, rois


def analyze_image_with_rois(image, num_divisions=50, roi_size=20, brightness_threshold=50):
    image_with_roi, rois = draw_rectangle_roi_base_on_points(image, num_divisions, roi_size)
    analysis_results = []

    for idx, roi in enumerate(rois):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi_corners = np.array(roi, dtype=np.int32)
        cv2.fillPoly(mask, [roi_corners], 255)
        roi_image = cv2.bitwise_and(image, image, mask=mask)
        roi_pixels = roi_image[mask == 255]

        if len(roi_pixels) == 0:
            continue

        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = round(np.mean(gray[mask == 255], dtype=float), 2)
        max_brightness = np.max(gray[mask == 255])
        min_brightness = np.min(gray[mask == 255])
        low_brightness_ratio = round(np.sum(gray[mask == 255] < brightness_threshold) / len(roi_pixels), 2)
        mean_color = np.mean(roi_pixels, axis=0).astype(int).tolist()

        analysis_results.append({
            'roi_id': idx + 1,
            'mean_brightness': mean_brightness,
            'max_brightness': max_brightness,
            'min_brightness': min_brightness,
            'mean_color': mean_color,
            'low_brightness_ratio': low_brightness_ratio
        })

    return image_with_roi, analysis_results

#################################################################################################


def detect_LED_blinking_from_camera(frame_queue, roi_size):
    roi_status = {}
    start_time = time.time()
    end_time = None

    while True:
        frame = frame_queue.get()
        _, analysis_results = analyze_image_with_rois(frame, roi_size=roi_size)

        current_time = time.time()
        on_rois = []
        unlit_rois = []
        for result in analysis_results:
            roi_id = result['roi_id']
            mean_brightness = result['mean_brightness']

            if mean_brightness > 50:
                on_rois.append(roi_id)
                if roi_id not in roi_status:
                    roi_status[roi_id] = 'on'
                    if all([roi_status.get(i, 'off') == 'on' for i in range(1, roi_id)]):
                        end_time = time.time()
                    else:
                        print(f"Error in lighting sequence at ROI {roi_id}")
            else:
                unlit_rois.append(roi_id)
                if roi_id in roi_status and roi_status[roi_id] == 'on':
                    print(f"Error: ROI {roi_id} turned off")

        all_on = all(status == 'on' for status in roi_status.values())
        if all_on:
            end_time = time.time()

        if end_time is not None:
            total_time = end_time - start_time
        else:
            total_time = current_time - start_time

        unlit_rois = [i for i in range(1, max(roi_status.keys()) + 1) if roi_status.get(i, 'off') == 'off']

        # 在图像上显示结果
        # cv2.putText(frame, f"Lighted ROI indexes: {on_rois}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(frame, f"Unlighted ROI indexes: {unlit_rois}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # cv2.putText(frame, f"Total time: {total_time:.2f} seconds", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #             (255, 255, 255), 2)

        # 定义每行文本的高度
        line_height = 30

        # 计算每个文本块的行数
        lighted_text_lines = f"Lighted ROI indexes:\n{on_rois}".split('\n')
        unlighted_text_lines = f"Unlighted ROI indexes:\n{unlit_rois}".split('\n')
        total_time_text_lines = f"Total time:\n{total_time:.2f} seconds".split('\n')

        # 绘制点亮的ROI索引
        y_start = 30
        for idx, line in enumerate(lighted_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制未点亮的ROI索引，调整起始位置以避免重叠
        y_start += len(lighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
        for idx, line in enumerate(unlighted_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 绘制总时间，调整起始位置以避免重叠
        y_start += len(unlighted_text_lines) * line_height + 10  # 10是两个块之间的间隔
        for idx, line in enumerate(total_time_text_lines):
            cv2.putText(frame, line, (10, y_start + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2)

        # 在终端上显示结果
        print(f"Lighted ROIs: {on_rois}")
        print(f"Unlit ROIs: {unlit_rois}")
        print(f"Total time: {total_time:.2f} seconds")

        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#########################################################################################
def main():
    # 初始化相机API
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()

    if dev_num == 0:
        print("没有检测到设备")
        return

    cam = device_manager.open_device_by_sn(dev_info_list[0].get("sn"))
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    cam.stream_on()

    frame_queue = queue.Queue(maxsize=100)

    capture_thread = threading.Thread(target=capture_frames, args=(cam, frame_queue))
    detect_thread = threading.Thread(target=detect_LED_blinking_from_camera, args=(frame_queue, 20))

    capture_thread.start()
    detect_thread.start()

    capture_thread.join()
    detect_thread.join()

    cam.stream_off()
    cam.close_device()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

