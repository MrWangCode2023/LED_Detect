import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from T1_contours import t1Contours


def t2ThinCurve(image, contour):
    # 创建一个与原图像相同尺寸的单通道空白图像
    contours_img = np.zeros(image.shape[:2], dtype=np.uint8)

    # 绘制轮廓
    cv2.drawContours(contours_img, [contour], -1, (255), -1)  # 轮廓颜色为255（白色）

    # 添加边框
    top, bottom, left, right = [2, 2, 2, 2]  # 边框大小
    color = [0]  # 黑色
    img = cv2.copyMakeBorder(contours_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # 细化算法进行细化
    skeleton_image = cv2.ximgproc.thinning(img)  # 细化
    skeleton = skeletonize(skeleton_image // 255)  # 转换为布尔型并细化
    thin_cnt_img = img_as_ubyte(skeleton)

    # 确保图像是二值图像
    if len(thin_cnt_img.shape) == 3:
        gray_image = cv2.cvtColor(thin_cnt_img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    else:
        binary_image = thin_cnt_img

    # 提取线条的坐标
    contour = np.column_stack(np.where(binary_image > 0))

    # 去除多余的维度
    if contour.ndim > 2:
        contour = np.squeeze(contour)

    # 确保轮廓点是二维的
    if contour.ndim == 1:
        contour = contour.reshape(-1, 2)

    # 显示结果
    cv2.imshow("Thin", thin_cnt_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return contour


if __name__ == "__main__":
    from T3_fitted_curve import t3FittedCurve

    # image = cv2.imread(r"E:\workspace\Data\LED_data\task1\3.bmp")
    image = cv2.imread(r"E:\workspace\Data\LED_data\task2\7.png")

    # 获取轮廓
    contours = t1Contours(image)
    for contour in contours:
        thin_contour = t2ThinCurve(image, contour)  # 处理每个轮廓
        # 这里可以对 thin_contour 进行进一步处理，例如拟合曲线
        # fitted_contour = t3FittedCurve(image, thin_contour)
