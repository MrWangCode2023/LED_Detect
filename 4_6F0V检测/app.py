import math
import cv2
from fractions import Fraction
from Object_extraction import object_extraction


def FOV_detect(image, f=18, f1=50, f2=5.359, w2=3, h2=2):
    """
    Args:
        image: 检测图像
        f: 光机距离
        f1: 相机距离
        f2: 焦距
        w2, h2: Sensor size
    Returns:
    """
    object_sizes = object_extraction(image)

    if not object_sizes:
        print("No objects detected.")
        return

    img_w1, img_h1 = object_sizes[0]  # 获取第一个对象的尺寸
    img_h2, img_w2, _ = image.shape  # 图像尺寸

    w1 = w2 * img_w1 / img_w2
    h1 = h2 * img_h1 / img_h2

    w = f1 * w1 / f2
    h = f1 * h1 / f2

    wfov = 2 * math.atan(w / (2 * f))
    hfov = 2 * math.atan(h / (2 * f))
    dfov = 2 * math.atan(math.sqrt(w ** 2 + h ** 2) / (2 * f))
    result0 = (wfov, hfov, dfov)

    WFOV = math.degrees(wfov)
    HFOV = math.degrees(hfov)
    DFOV = math.degrees(dfov)
    result1 = (WFOV, HFOV, DFOV)

    # 将弧度结果转换为分数
    wfov_fraction = Fraction(wfov).limit_denominator()
    hfov_fraction = Fraction(hfov).limit_denominator()
    dfov_fraction = Fraction(dfov).limit_denominator()
    result2 = (wfov_fraction, hfov_fraction, dfov_fraction)

    # 打印结果
    print(f"\n| 角度数据| WFOV：{WFOV:.4f} | HFOV:{HFOV:.4f} | DFOV:{DFOV:.4f} |")
    print(f"| 弧度角度| WFOV：{wfov:.4f} π | HFOV:{hfov:.4f} π | DFOV:{dfov:.4f} π |")
    print(f"| 弧度分式| WFOV：{wfov_fraction} π | HFOV:{hfov_fraction} π | DFOV:{dfov_fraction} π |")

    result = {
        "角度数据": result0,
        "弧度角度": result1,
        "弧度分式": result2
    }

    return result


if __name__ == "__main__":
    image = cv2.imread(r"E:\workspace\Data\LED_data\task4\11.bmp")
    result = FOV_detect(image)
