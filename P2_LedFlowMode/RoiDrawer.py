import numpy as np
import cv2

class ShapeDrawer:
    def __init__(self, shape_type, data):
        """
        初始化 ShapeDrawer 类。

        参数：
        - shape_type: 形状类型 ('contour', 'center', 'box')
        - data: 形状数据，根据 shape_type 的不同类型而不同
        """
        self.shape_type = shape_type
        self.data = np.array(data) if data is not None else None

    def draw_roi(self, image, color=(0, 255, 0), thickness=1):
        """
        根据 shape_type 和数据绘制 ROI 区域。

        参数：
        - image: 要绘制的图像
        - color: 绘制 ROI 的颜色，默认为绿色 (0, 255, 0)
        - thickness: 绘制 ROI 的线条厚度，默认为 2

        返回：
        - image: 绘制 ROI 后的图像
        """
        if self.shape_type == 'contour' and self.data is not None:
            contour = self.data.astype(int)
            cv2.drawContours(image, [contour], -1, color, thickness)
        elif self.shape_type == 'center' and self.data is not None:
            center = tuple(self.data.astype(int))
            cv2.circle(image, center, 10, color, -1)
        elif self.shape_type == 'box' and self.data is not None:
            box = self.data.astype(int)
            cv2.drawContours(image, [box], 0, (255, 255, 255), 1)
        else:
            raise ValueError("无效的 shape_type 或数据为空")
        return image

# 示例使用
if __name__ == "__main__":
    # 创建一个空白图像
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    # 示例数据
    contour = [[100, 100], [200, 100], [200, 200], [100, 200]]
    center = [300, 300]
    box = [[400, 400], [500, 400], [500, 500], [400, 500]]

    # 使用轮廓数据
    drawer = ShapeDrawer('contour', contour)
    image_with_contour = drawer.draw_roi(image.copy())
    cv2.imshow("Contour", image_with_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 使用中心点数据
    drawer = ShapeDrawer('center', center)
    image_with_center = drawer.draw_roi(image.copy())
    cv2.imshow("Center", image_with_center)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 使用矩形框数据
    drawer = ShapeDrawer('box', box)
    image_with_box = drawer.draw_roi(image.copy())
    cv2.imshow("Box", image_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
