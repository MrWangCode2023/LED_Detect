import cv2
import numpy as np


def find_max_luminance_region(img, luminance_image, step=1):
    def luminance_to_illuminance(l, k=0.1):
        return k * l
    # Convert to grayscale if the image is colored
    # show_image = luminance_image.copy()
    if len(luminance_image.shape) == 3:
        luminance_image = cv2.cvtColor(luminance_image, cv2.COLOR_BGR2GRAY)

    luminance_image = luminance_image.astype(np.float32)

    # Initialize variables
    optimal_binary = None
    num_regions = 0
    max_luminance_mean = -np.inf
    max_luminance_centroid = None
    regions_info = []

    # Calculate the maximum and minimum values in the image
    max_value = np.max(luminance_image)
    min_value = np.min(luminance_image)
    optimal_threshold = max_value

    # Iterate from max value to min value
    for threshold in np.arange(max_value, min_value, -step):
        # Apply binary thresholding
        _, binary_image = cv2.threshold(luminance_image, threshold, 255, cv2.THRESH_BINARY)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8), connectivity=8)

        # Calculate the number of regions (excluding background)
        num_regions = num_labels - 1

        if num_regions > 0:
            # If there are regions, update the optimal threshold and region
            optimal_threshold = threshold
            optimal_binary = binary_image

            # Calculate information for each region
            for label in range(1, num_labels):
                mask = (labels == label)
                region_luminance_mean = np.mean(luminance_image[mask])  # 区域亮度均值

                # Check if this region has the maximum luminance mean
                if region_luminance_mean > max_luminance_mean:
                    max_luminance_mean = region_luminance_mean  # 最大亮度均值
                    max_luminance_centroid = tuple(map(int, centroids[label]))  # 最大亮度均值坐标

                luminance_value = luminance_image[int(centroids[label][1]), int(centroids[label][0])]
                region_info = {
                    "编号": label,
                    "质心坐标": tuple(map(int, centroids[label])),
                    "质心照度值": luminance_to_illuminance(luminance_value),
                    "区域照度均值": luminance_to_illuminance(region_luminance_mean),
                    "区域面积": stats[label, cv2.CC_STAT_AREA],
                }
                # luminance_to_illuminance()
                regions_info.append(region_info)
            break  # Exit the loop after finding the first set of regions

    if regions_info:
        print(f"照度区域信息表：")
        for region_info in regions_info:
            print(f"| 区域编号: {region_info['编号']} | Max照度值坐标: {region_info['质心坐标']} | Max照度值: {region_info['质心照度值']} | 区域照度均值: {region_info['区域照度均值']} | 区域面积: {region_info['区域面积']} |\n\n")

    # 基于质心画roi
    max_illuminance_value = luminance_to_illuminance(max_luminance_mean)
    Emax = {"Emax坐标": max_luminance_centroid,
            "Emax值": max_illuminance_value}
    print(f"照度区域数量: {num_regions}\nEmax坐标：{max_luminance_centroid}\nEmax值：{max_illuminance_value}")

    show_image = img.copy()
    cv2.circle(show_image, max_luminance_centroid, 5, (0, 0, 255), -1)
    result = [Emax, regions_info, show_image, optimal_binary]

    return result


if __name__ == "__main__":
    # 示例使用
    image_path = r"E:\workspace\Data\LED_data\task4\2.bmp"
    illuminance_image = cv2.imread(image_path)

    if illuminance_image is None:
        raise FileNotFoundError("The specified image file could not be found or read.")

    result = find_max_luminance_region(illuminance_image)
