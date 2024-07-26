import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from ImagePreprocessor import ImagePreprocessor

# TextureFeatureExtractor 类的修改
from skimage.feature import local_binary_pattern


# 纹理特征
class TextureFeatureExtractor(ImagePreprocessor):
    def extract_texture_features(self):
        self.BGR2GRAY()
        self.CLAHE()
        # 使用skimage的LBP（局部二值模式）提取纹理特征
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(self.image, n_points, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                                 range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # 归一化
        return hist




# 形状特征
class ShapeFeatureExtractor(ImagePreprocessor):
    def extract_shape_features(self):
        self.BGR2GRAY()
        self.GaussianBlur((5, 5))
        contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            shapes.append(approx)
        return shapes

# 颜色特征
class ColorFeatureExtractor(ImagePreprocessor):
    def extract_color_histogram(self):
        chans = cv2.split(self.image)
        colors = ("b", "g", "r")
        features = []
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist.flatten())
        return np.array(features)


# 边缘特征
class EdgeFeatureExtractor(ImagePreprocessor):
    def extract_edge_features(self):
        self.BGR2GRAY()
        self.GaussianBlur((5, 5))
        edges = self.Edges(20, 60)
        return edges


# 频域特征
class FrequencyFeatureExtractor(ImagePreprocessor):
    def extract_frequency_features(self):
        self.BGR2GRAY()
        dft = cv2.dft(np.float32(self.image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return magnitude_spectrum

# 统计特征
class StatisticalFeatureExtractor(ImagePreprocessor):
    def extract_statistical_features(self):
        mean, std_dev = cv2.meanStdDev(self.image)
        return mean, std_dev

# 关键点和局部描述子
class KeypointFeatureExtractor(ImagePreprocessor):
    def extract_keypoints_and_descriptors(self):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.image, None)
        return keypoints, descriptors


if __name__ == "__main__":
    image_path = 'E:\workspace\Data\LED_data\custom\LED3.jpg'
    image = cv2.imread(image_path)

    # 使用纹理特征提取器
    texture_extractor = TextureFeatureExtractor(image)
    texture_hist = texture_extractor.extract_texture_features()

    # 使用形状特征提取器
    shape_extractor = ShapeFeatureExtractor(image)
    shapes = shape_extractor.extract_shape_features()

    # 使用颜色特征提取器
    color_extractor = ColorFeatureExtractor(image)
    color_histogram = color_extractor.extract_color_histogram()

    # 使用边缘特征提取器
    edge_extractor = EdgeFeatureExtractor(image)
    edges = edge_extractor.extract_edge_features()

    # 使用频域特征提取器
    frequency_extractor = FrequencyFeatureExtractor(image)
    frequency_features = frequency_extractor.extract_frequency_features()

    # 使用统计特征提取器
    statistical_extractor = StatisticalFeatureExtractor(image)
    mean, std_dev = statistical_extractor.extract_statistical_features()

    # 使用关键点和局部描述子提取器
    keypoint_extractor = KeypointFeatureExtractor(image)
    keypoints, descriptors = keypoint_extractor.extract_keypoints_and_descriptors()

    # 显示结果
    # texture_extractor.display_image("Texture Features")
    # shape_extractor.display_image("Shape Features")
    # color_extractor.display_image("Color Histogram")
    edge_extractor.display_image("Edge Features")
    # frequency_extractor.display_image("Frequency Features")i
    # statistical_extractor.display_image("Statistical Features")
    # keypoint_extractor.display_image("Keypoints and Descriptors")


