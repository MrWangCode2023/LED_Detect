import numpy as np
import cv2
import matplotlib.pyplot as plt
from FeatureExtraction import TextureFeatureExtractor, ShapeFeatureExtractor, ColorFeatureExtractor, EdgeFeatureExtractor, FrequencyFeatureExtractor, StatisticalFeatureExtractor, KeypointFeatureExtractor

class FeatureFusion:
    def __init__(self, image, features_to_extract, keypoint_descriptor_length=128, max_keypoints=70):
        self.image = image
        self.features_to_extract = features_to_extract
        self.keypoint_descriptor_length = keypoint_descriptor_length
        self.max_keypoints = max_keypoints
        self.visualization_data = []  # Initialize list to store visualization data

    def extract_features(self):
        features = {}

        if 'texture' in self.features_to_extract:
            texture_extractor = TextureFeatureExtractor(self.image)
            texture_hist = texture_extractor.extract_texture_features()
            features['texture'] = texture_hist
            self.visualize_feature(texture_hist, 'Texture')

        if 'shape' in self.features_to_extract:
            shape_extractor = ShapeFeatureExtractor(self.image)
            shapes = shape_extractor.extract_shape_features()
            shape_image = np.zeros_like(self.image)
            cv2.drawContours(shape_image, shapes, -1, (255, 255, 255), 2)
            features['shape'] = shape_image
            self.visualize_feature(shape_image, 'Shape', is_image=True)

        if 'color' in self.features_to_extract:
            color_extractor = ColorFeatureExtractor(self.image)
            color_histogram = color_extractor.extract_color_histogram()
            features['color'] = color_histogram
            self.visualize_feature(color_histogram, 'Color')

        if 'edge' in self.features_to_extract:
            edge_extractor = EdgeFeatureExtractor(self.image)
            edges = edge_extractor.extract_edge_features()
            features['edge'] = edges
            self.visualize_feature(edges, 'Edge', is_image=True)

        if 'frequency' in self.features_to_extract:
            frequency_extractor = FrequencyFeatureExtractor(self.image)
            frequency_features = frequency_extractor.extract_frequency_features()
            features['frequency'] = frequency_features
            self.visualize_feature(frequency_features.flatten(), 'Frequency')

        if 'statistical' in self.features_to_extract:
            statistical_extractor = StatisticalFeatureExtractor(self.image)
            mean, std_dev = statistical_extractor.extract_statistical_features()
            features['statistical'] = (mean, std_dev)
            self.visualize_feature(mean.flatten(), 'Statistical Mean')
            self.visualize_feature(std_dev.flatten(), 'Statistical Std Dev')

        if 'keypoint' in self.features_to_extract:
            keypoint_extractor = KeypointFeatureExtractor(self.image)
            keypoints, descriptors = keypoint_extractor.extract_keypoints_and_descriptors()
            keypoint_image = self.image.copy()
            cv2.drawKeypoints(self.image, keypoints, keypoint_image)
            features['keypoint'] = keypoint_image
            self.visualize_feature(keypoint_image, 'Keypoint', is_image=True)

        return features

    def visualize_feature(self, feature, title, is_image=False):
        """
        可视化特征，将特征向量转换为可视化的图像形式。
        """
        if not is_image:
            feature_length = len(feature)
            feature_size = int(np.ceil(np.sqrt(feature_length)))
            feature_image = np.zeros((feature_size, feature_size), dtype=np.uint8)
            feature_image = feature_image.flatten()
            feature_image[:feature_length] = feature

            # Resize to original image size for better visualization
            feature_image = feature_image.reshape((feature_size, feature_size))
            feature_image = cv2.resize(feature_image, (256, 256), interpolation=cv2.INTER_NEAREST)
            feature_image = cv2.normalize(feature_image, None, 0, 255, cv2.NORM_MINMAX)
        else:
            feature_image = feature

        # Append the feature image and title to the list for later visualization
        self.visualization_data.append((feature_image, title))

    def save_features(self, save_path):
        """
        保存所有特征图像。
        """
        for idx, (feature_image, title) in enumerate(self.visualization_data):
            if feature_image.ndim == 2:
                cv2.imwrite(f"{save_path}/{title}.png", feature_image)
            else:
                cv2.imwrite(f"{save_path}/{title}.png", cv2.cvtColor(feature_image, cv2.COLOR_BGR2RGB))

    def show_features(self):
        """
        显示所有特征图像。
        """
        num_features = len(self.visualization_data)
        cols = 3
        rows = (num_features + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axs = axs.flatten()

        for idx, (feature_image, title) in enumerate(self.visualization_data):
            if feature_image.ndim == 2:
                axs[idx].imshow(feature_image, cmap='gray')
            else:
                axs[idx].imshow(cv2.cvtColor(feature_image, cv2.COLOR_BGR2RGB))
            axs[idx].set_title(title)
            axs[idx].axis('off')

        for idx in range(len(self.visualization_data), len(axs)):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    image_path = 'E:\\workspace\\Data\\LED_data\\4_11\\1.bmp'
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))

    # 选择要提取的特征
    features_to_extract = ['texture', 'shape', 'color', 'edge', 'frequency', 'statistical', 'keypoint']

    # 实例化FeatureFusion类并提取特征
    feature_fusion = FeatureFusion(image, features_to_extract)
    features = feature_fusion.extract_features()

    # 显示特征图像
    feature_fusion.show_features()

    # 保存特征图像
    save_path = 'E:\\workspace\\Data\\LED_data\\4_11\\features'
    feature_fusion.save_features(save_path)
