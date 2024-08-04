import cv2
import numpy as np
from t1_edges import t1_edges
from t2_curvatureCalculate import t2_curvatureCalculate
from t3_getLeftEdge import t3_getLeftEdgePoints
from t4_left_edge_mid_point import t4_left_edge_mid_point
from t5_drawCoordinateSystem import t5_drawCoordinateSystem
from t7_mappingMatrix import T7_mappingMatrix
from t8_coordinateTransformation import T8_coordinateTransformation
from t10_saveParameter2Json import t10_saveParameter2Json


def A0_HV2IlluminanceMapping(image, point_illuminances, degree=2):
    """
    拟合亮度值与照度值之间的映射关系。

    Args:
        image: 输入图像
        point_illuminances: [[x坐标， y坐标， 照度值], ...]
        degree: 拟合多项式的阶数

    Returns:
        T: 坐标系映射矩阵
        T_inv: 逆映射矩阵
        coefficients: 拟合的多项式系数
        p0: 原点
        p1: 边缘均值点
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 建立坐标系变换映射矩阵
    edge_coordinates, edges_image = t1_edges(gray)

    # 计算曲率并找到最大曲率的点作为原点
    p0 = t2_curvatureCalculate(edge_coordinates)

    # 获得拐点左下部分边缘线的坐标
    left_edge_points = t3_getLeftEdgePoints(edge_coordinates, p0)

    # 计算边缘均值点p1
    p1 = t4_left_edge_mid_point(left_edge_points)

    # 建立两个坐标系的映射关系矩阵
    T, T_inv = T7_mappingMatrix(p0, p1)

    # 2. 建立亮度——照度映射关系
    luminances, illuminances = [], []

    for point_illuminance in point_illuminances:
        if len(point_illuminance) != 3:
            raise ValueError("每个点的格式应为 [x, y, illuminance]。")

        x, y, illuminance = point_illuminance

        if illuminance < 0:
            raise ValueError("照度值必须为正数。")

        if T_inv is not None:
            # 坐标系转换
            w, h = T8_coordinateTransformation((x, y), T_inv)

            if 0 <= int(h) < gray.shape[0] and 0 <= int(w) < gray.shape[1]:
                luminance = gray[int(h), int(w)]
            else:
                luminance = 0  # 如果超出范围，使用默认值
        else:
            luminance = 0  # 如果没有 T_inv，使用默认值

        luminances.append(luminance)
        illuminances.append(illuminance)

    # 确保输入是 NumPy 数组
    luminances = np.array(luminances)
    illuminances = np.array(illuminances)

    # 确保有有效的亮度数据
    if len(luminances) == 0 or len(illuminances) == 0:
        raise ValueError("没有有效的亮度或照度值进行拟合。")

    # 使用多项式拟合，返回多项式的系数
    coefficients = np.polyfit(luminances, illuminances, degree)

    # 保存结果到 parameter.json 中
    t10_saveParameter2Json("parameter.json", T, T_inv, coefficients, p0, p1)

    # 可视化
    show_img = image.copy()
    t5_drawCoordinateSystem(show_img, p0, p1)

    # 绘制边缘线和照度点
    for p, illuminance in zip(edge_coordinates, illuminances):
        cv2.circle(show_img, (p[0], p[1]), 1, (255, 255, 255), -1)  # 绘制边缘点
        # 显示照度值
        # cv2.putText(show_img, f"{illuminance}", (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 绘制原点
    cv2.circle(show_img, (int(p0[0]), int(p0[1])), 3, (0, 0, 255), -1)  # 红色点

    # 显示图像
    cv2.imshow('show_img', show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return T, T_inv, coefficients, p0, p1


# 测试部分
if __name__ == "__main__":
    from t11_loadParameterFromJson import t11_loadParameterFromJson

    # 读取灰度图像
    image = cv2.imread('../../../projectData/LED_data/task4/2.bmp')
    if image is None:
        raise ValueError("无法读取图像，请检查文件路径。")

    # 模拟照度数据，格式为 [[x, y, illuminance], ...]
    point_illuminances = [
        [10, 20, 300],
        [50, 60, 350],
        [100, 120, 400],
        [150, 180, 450],
        [200, 220, 500],
        # 可以添加更多数据点
    ]

    # 调用函数进行拟合
    T, T_inv, coefficients, p0, p1 = A0_HV2IlluminanceMapping(image, point_illuminances, degree=2)

    # 输出结果
    print("映射矩阵 T:\n", T)
    print("逆映射矩阵 T_inv:\n", T_inv)
    print("拟合的多项式系数:\n", coefficients)

    # 从 JSON 文件读取
    T_loaded, T_inv_loaded, coefficients_loaded, p0_loaded, p1_loaded = t11_loadParameterFromJson('parameter.json')

    # 输出读取的数据
    print("从 JSON 文件读取的映射矩阵 T:\n", T_loaded)
    print("从 JSON 文件读取的逆映射矩阵 T_inv:\n", T_inv_loaded)
    print("从 JSON 文件读取的拟合的多项式系数:\n", coefficients_loaded)
    print("从 JSON 文件读取的原点 p0:\n", p0_loaded)
    print("从 JSON 文件读取的边缘均值点 p1:\n", p1_loaded)
