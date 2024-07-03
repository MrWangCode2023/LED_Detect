import cv2
import numpy as np

def edge_center_to_vertext(E, F, G, H, image):
    A1_4image = image.copy()
    result = []

    xe, ye = E
    xf, yf = F
    xg, yg = G
    xh, yh = H

    k1 = (ye - yg) / (xe - xg)
    k2 = (yf - yh) / (xf - xh)

    # A点坐标
    xa = (k1 * xf - yf - k2 * xe + ye) / (k1 - k2)
    ya = k1 * (xa - xf) + yf

    # B点坐标
    xb = (k1 * xf - yf - k2 * xg + yg) / (k1 - k2)
    yb = k1 * (xb - xf) + yf

    # C点坐标
    xc = (k1 * xh - yh - k2 * xg + yg) / (k1 - k2)
    yc = k1 * (xc - xh) + yh

    # D点坐标
    xd = (k1 * xh - yh -k2 * xe + ye) / (k1 - k2)
    yd = k1 * (xd - xh) + yh

    A1 = (int(xa), int(ya))
    B1 = (int(xb), int(yb))
    C1 = (int(xc), int(yc))
    D1 = (int(xd), int(yd))

    result = [A1, B1, C1, D1]
    # result = {
    #     "A1": A1,
    #     "B1": B1,
    #     "C1": C1,
    #     "D1": D1,
    # }
    # # print("顶点坐标：", result)
    #
    # # 在图像中标记顶点坐标并显示名称
    # for name, point in result.items():
    #     cv2.circle(A1_4image, point, 5, (0, 0, 255), -1)
    #     cv2.putText(A1_4image, name, (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    #
    # # 绘制矩形
    # pts = np.array([A1, B1, C1, D1], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # cv2.polylines(A1_4image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)


    return result

if __name__ == "__main__":
    E = (125, 231)
    F = (394, 120)
    G = (589, 338)
    H = (316, 456)


    image = cv2.imread("E:\\workspace\\Data\\LED_data\\task4\\19.png")

    if image is None:
        print("Error: Image not found or unable to open.")
    else:
        result = edge_center_to_vertext(E, F, G, H, image)

