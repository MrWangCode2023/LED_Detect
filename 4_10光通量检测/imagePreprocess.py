import cv2
import numpy as np
from customRoiMask import CustomRoiMask


def image_preprocess(image):
    # 确保图像已正确加载
    if image is None:
        raise ValueError("Image not loaded")

    # 计算灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将灰度图保持为单通道
    # gray的形状应为 (height, width)

    # 合并BGR图像和灰度图
    merged = np.concatenate((image, gray[:, :, np.newaxis]), axis=2)  # 形状为 (height, width, 4)

    h, w, _ = image.shape
    mask = CustomRoiMask(h, w, [(0.5, 0.5)], radius=None)

    # 使用位与操作
    roi_img = cv2.bitwise_and(gray, mask)  # 确保gray和mask形状一致

    # 在此处不需要增加roi_img的通道维度
    # 直接合并roi_img到merged
    merged = np.concatenate((merged, roi_img[:, :, np.newaxis]), axis=2)

    # 打印形状信息用于调试
    # print(f"gray shape: {gray.shape}")
    # print(f"oi_img shape: {roi_img.shape}")
    # print(f"merged shape: {merged.shape}")

    return merged


if __name__ == "__main__":
    # 示例使用
    image = cv2.imread('../../projectData/LED_data/task4/2.bmp')
    bgr_gray_roiImg = image_preprocess(image)

    # 显示图像
    cv2.imshow("image", bgr_gray_roiImg[:, :, 4])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
