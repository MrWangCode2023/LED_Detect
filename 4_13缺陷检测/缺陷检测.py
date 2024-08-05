import cv2
import numpy as np

class TraditionalDefectDetector:
    def __init__(self):
        pass

    def preprocess_image(self, image_path, target_size=(256, 256)):
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)  # 调整图像大小
        return image

    def detect_edges(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)
        return edges

    def detect_contours(self, edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def extract_shape_features(self, contours):
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 忽略小的噪声轮廓
                shapes.append(contour)
        return shapes

    def extract_color_features(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([image_hsv], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([image_hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([image_hsv], [2], None, [256], [0, 256])
        return np.concatenate((h_hist, s_hist, v_hist)).flatten()

    def extract_texture_features(self, image):
        glcm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_image = glcm.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        return cl_image.flatten()

    def detect_defects(self, image, contours, texture_features, color_features):
        defects = []
        defect_contours = []

        # 示例规则1：轮廓面积过大或过小
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000 or area < 100:
                defects.append(f'Area defect: {area}')
                defect_contours.append(contour)

        # 示例规则2：颜色直方图均值过高或过低
        color_mean = np.mean(color_features)
        if color_mean > 0.5 or color_mean < 0.2:
            defects.append(f'Color defect: mean {color_mean}')

        # 示例规则3：纹理特征均值过高或过低
        texture_mean = np.mean(texture_features)
        if texture_mean > 0.5 or texture_mean < 0.2:
            defects.append(f'Texture defect: mean {texture_mean}')

        # 在图像中绘制检测到的缺陷轮廓
        defect_image = image.copy()
        cv2.drawContours(defect_image, defect_contours, -1, (0, 0, 255), 2)

        return defects, defect_image

    def detect(self, image_path):
        image = self.preprocess_image(image_path)
        edges = self.detect_edges(image)
        contours = self.detect_contours(edges)
        shapes = self.extract_shape_features(contours)
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)

        defects, defect_image = self.detect_defects(image, shapes, texture_features, color_features)
        return defects, defect_image

if __name__ == "__main__":
    detector = TraditionalDefectDetector()

    train_image_path = 'E:\workspace\Data\LED_data\\4_11\\1.bmp'
    train_defects, train_defect_image = detector.detect(train_image_path)
    print(f"训练图像缺陷检测结果: {train_defects}")
    cv2.imshow('Train Image Defects', train_defect_image)
    cv2.imwrite('train_image_defects.png', train_defect_image)

    new_image_path = 'E:\workspace\Data\LED_data\\4_11\\1.bmp'
    new_defects, new_defect_image = detector.detect(new_image_path)
    print(f"新图像缺陷检测结果: {new_defects}")
    cv2.imshow('New Image Defects', new_defect_image)
    cv2.imwrite('new_image_defects.png', new_defect_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
