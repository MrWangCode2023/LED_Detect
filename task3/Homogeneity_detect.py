import cv2
from common.Common import analyze_image_with_rois


def homogeneity_detect(image_path):
    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    if image is None:
        raise ValueError(f"图像加载失败: {image_path}")

    # 绘制ROI并分析
    image_with_roi, analysis_results = analyze_image_with_rois(image, num_divisions=30, roi_size=20, brightness_threshold=50)

    # 打印结果
    for result in analysis_results:
        print(result)

    # 显示结果图像
    cv2.imshow('Image with ROIS', image_with_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows("q")


if __name__ == '__main__':
    image_path = '../../Data/LED_data/task1/task1_11.bmp'
    homogeneity_detect(image_path)
