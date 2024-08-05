import cv2
import numpy as np
from Show import show_image
from t1_sortedContours import t1_sortedContours
from t2_roiPixelsImg import t2_roiPixelsImg


def APP(image):
    # 1 添加边框
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 在图像上绘制边框
    border_size = 1
    img2 = cv2.rectangle(image, (0, 0), (width, height), (0, 0, 0), border_size)

    results = []

    # 2 图像预处理
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    blured = cv2.GaussianBlur(closed, (5, 5), 0)

    # 3 获取排序后的轮廓
    contours_sorted = t1_sortedContours(blured)

    mask1 = np.zeros(gray.shape, dtype=np.uint8)  # mask 应为单通道，大小与 gray 一致

    # 5. 处理所有轮廓
    for idx, cnt in enumerate(contours_sorted[1:-1]):
        # 4. 创建一个与原图像大小一致的空白单通道 mask
        mask = np.zeros(gray.shape, dtype=np.uint8)  # mask 应为单通道，大小与 gray 一致

        # 5. 在掩码上填充当前轮廓
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        # cv2.drawContours(mask1, [cnt], -1, 255, thickness=1)
        # 绘制轮廓，使用红色（BGR格式）
        cv2.drawContours(image, [cnt], -1, (0, 0, 255), thickness=1)  # 红色

        # 6. 确保掩码与原图像大小一致
        contour_pixels = cv2.bitwise_and(img2, img2, mask=mask)

        # 7. 计算区域像素值
        result = t2_roiPixelsImg(contour_pixels, darkThreshold=50)
        result.insert(0, idx)
        results.append(result)

    results = np.array(results)

    return results, image


if __name__ == "__main__":
    # 读取图像
    image = cv2.imread(r"../../../projectData/LED_data/4_11/3.bmp")
    results, mask1 = APP(image)

    print(f"results: {results}")

    # 显示结果
    cv2.imshow('Original Image', image)
    # cv2.imshow('Contour Pixels', contour_pixels)
    cv2.imshow('Mask', mask1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
