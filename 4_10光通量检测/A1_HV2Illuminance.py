import numpy as np
import cv2
from t5_drawCoordinateSystem import t5_drawCoordinateSystem
from t8_coordinateTransformation import T8_coordinateTransformation
from t9_luminance2illuminance import t9_luminance2illuminance


def detect(image, points, T, T_inv, coefficients, p0, p1):
    """
    计算图像中指定检测点的照度值。

    Args:
        image: 输入图像
        points: 检测点位，格式为 [[x1, y1], [x2, y2], ...]
        T: 坐标系映射矩阵
        T_inv: 逆坐标转换矩阵
        coefficients: 亮度-照度映射系数

    Returns:
        illuminances: 计算得到的照度值
    """
    show_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    illuminances = []

    for p in points:
        # 新坐标系坐标转映射成图像坐标系
        pt = T8_coordinateTransformation(p, T_inv)

        # 转换为整数索引
        x, y = int(pt[0]), int(pt[1])

        # 确保坐标在图像范围内
        if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
            # 计算当前点的亮度
            luminance = gray[y, x]
            # 计算照度值
            illuminance = t9_luminance2illuminance(luminance, coefficients)
        else:
            # 如果坐标超出范围，使用默认值
            illuminance = 0

        illuminances.append(illuminance)

        # 在图像上绘制检测点
        cv2.circle(show_img, (x, y), 3, (0, 255, 0), -1)  # 绿色点
        cv2.putText(show_img, f"{illuminance:.2f}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # 显示照度值

    # 可视化
    # 5 在原图上绘制坐标系
    t5_drawCoordinateSystem(show_img, p0, p1)
    # 绘制原点
    cv2.circle(show_img, (int(p0[0]), int(p0[1])), 3, (0, 0, 255), -1)  # 红色点

    # 显示图像
    cv2.imshow('show_img', show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return illuminances


if __name__ == "__main__":
    from t11_loadParameterFromJson import t11_loadParameterFromJson

    # 读取灰度图像
    image = cv2.imread('../../../projectData/LED_data/task4/2.bmp')
    if image is None:
        raise ValueError("无法读取图像，请检查文件路径。")

    # 测试检测点
    points = [
        [10, 20],
        [50, 60],
        [100, 120],
        [150, 180],
        [200, 220]
    ]

    # 从 JSON 文件加载参数
    T, T_inv, coefficients, p0, p1 = t11_loadParameterFromJson("parameter.json")

    # 调用检测函数
    illuminances = detect(image, points, T, T_inv, coefficients, p0, p1)

    # 输出计算得到的照度值
    for point, illuminance in zip(points, illuminances):
        print(f"点 {point} 的照度值: {illuminance}")
