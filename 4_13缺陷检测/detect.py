import numpy as np
import cv2
from FeatureExtraction import TextureFeatureExtractor, ShapeFeatureExtractor, ColorFeatureExtractor, EdgeFeatureExtractor, FrequencyFeatureExtractor, StatisticalFeatureExtractor, KeypointFeatureExtractor

class FeatureFusion:
    def __init__(self, image, features_to_extract, keypoint_descriptor_length=128, max_keypoints=70):
        self.image = image
        self.features_to_extract = features_to_extract
        self.keypoint_descriptor_length = keypoint_descriptor_length
        self.max_keypoints = max_keypoints

    def extract_features(self):
        features = []

        if 'texture' in self.features_to_extract:
            texture_extractor = TextureFeatureExtractor(self.image)
            texture_hist = texture_extractor.extract_texture_features()
            features.append(texture_hist)
            print(f"纹理特征维度: {texture_hist.shape}")

        if 'shape' in self.features_to_extract:
            shape_extractor = ShapeFeatureExtractor(self.image)
            shapes = shape_extractor.extract_shape_features()
            features.append(np.array([len(shapes)]))  # 用形状的数量作为特征
            print(f"形状特征维度: {np.array([len(shapes)]).shape}")

        if 'color' in self.features_to_extract:
            color_extractor = ColorFeatureExtractor(self.image)
            color_histogram = color_extractor.extract_color_histogram()
            features.append(color_histogram)
            print(f"颜色特征维度: {color_histogram.shape}")

        if 'edge' in self.features_to_extract:
            edge_extractor = EdgeFeatureExtractor(self.image)
            edges = edge_extractor.extract_edge_features()
            features.append(edges.flatten())
            print(f"边缘特征维度: {edges.flatten().shape}")

        if 'frequency' in self.features_to_extract:
            frequency_extractor = FrequencyFeatureExtractor(self.image)
            frequency_features = frequency_extractor.extract_frequency_features()
            features.append(frequency_features.flatten())
            print(f"频率特征维度: {frequency_features.flatten().shape}")

        if 'statistical' in self.features_to_extract:
            statistical_extractor = StatisticalFeatureExtractor(self.image)
            mean, std_dev = statistical_extractor.extract_statistical_features()
            features.append(np.concatenate((mean.flatten(), std_dev.flatten())))
            print(f"统计特征维度: {np.concatenate((mean.flatten(), std_dev.flatten())).shape}")

        if 'keypoint' in self.features_to_extract:
            keypoint_extractor = KeypointFeatureExtractor(self.image)
            keypoints, descriptors = keypoint_extractor.extract_keypoints_and_descriptors()
            if descriptors is not None:
                if descriptors.shape[0] < self.max_keypoints:
                    padded_descriptors = np.zeros((self.max_keypoints, self.keypoint_descriptor_length), dtype=np.float32)
                    padded_descriptors[:descriptors.shape[0], :] = descriptors
                else:
                    padded_descriptors = descriptors[:self.max_keypoints, :]
                features.append(padded_descriptors.flatten())
                print(f"关键点特征维度: {padded_descriptors.flatten().shape}")
            else:
                features.append(np.zeros((self.max_keypoints * self.keypoint_descriptor_length,), dtype=np.float32))
                print(f"关键点特征维度: {np.zeros((self.max_keypoints * self.keypoint_descriptor_length,), dtype=np.float32).shape}")

        # 将所有特征融合成一个特征向量
        fused_features = np.concatenate(features, axis=0)
        return fused_features

class RuleBasedAnomalyDetector:
    def __init__(self, rules):
        self.rules = rules

    def detect(self, features):
        for rule in self.rules:
            if not rule(features):
                return "异常"
        return "正常"

def preprocess_image(image_path, target_size=(256, 256)):
    """
    预处理图像，将图像调整为目标大小。
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)  # 调整图像大小
    return image

# 定义检测规则
def rule_texture(features):
    texture_features = features[:10]
    return np.mean(texture_features) < 0.5  # 示例规则：纹理特征均值小于0.5

def rule_shape(features):
    shape_features = features[10:11]
    return shape_features[0] > 5  # 示例规则：形状特征数量大于5

def rule_color(features):
    color_features = features[11:779]
    return np.mean(color_features) < 0.3  # 示例规则：颜色特征均值小于0.3

def rule_edge(features):
    edge_features = features[779:66315]
    return np.mean(edge_features) < 0.4  # 示例规则：边缘特征均值小于0.4

def rule_frequency(features):
    frequency_features = features[66315:131091]
    return np.mean(frequency_features) < 0.5  # 示例规则：频率特征均值小于0.5

def rule_statistical(features):
    statistical_features = features[131091:131097]
    return np.mean(statistical_features) < 0.2  # 示例规则：统计特征均值小于0.2

def rule_keypoint(features):
    keypoint_features = features[131097:]
    return np.mean(keypoint_features) < 0.6  # 示例规则：关键点特征均值小于0.6

# 主程序
if __name__ == "__main__":
    image_path = 'E:\\workspace\\Data\\LED_data\\task1\\task1_5.bmp'
    image = preprocess_image(image_path)
    print(f"训练图像尺寸: {image.shape}")

    # 提取特征并进行融合
    feature_fusion = FeatureFusion(image, features_to_extract=['texture', 'shape', 'color', 'edge', 'frequency', 'statistical', 'keypoint'])
    fused_feature = feature_fusion.extract_features()
    print(f"训练图像特征维度: {fused_feature.shape}")

    # 检测规则
    rules = [rule_texture, rule_shape, rule_color, rule_edge, rule_frequency, rule_statistical, rule_keypoint]

    # 训练检测模型
    detector = RuleBasedAnomalyDetector(rules)

    # 检测新图像
    new_image_path = 'E:\\workspace\\Data\\LED_data\\task1\\task1_6.bmp'
    new_image = preprocess_image(new_image_path)
    print(f"新图像尺寸: {new_image.shape}")

    new_feature_fusion = FeatureFusion(new_image, features_to_extract=['texture', 'shape', 'color', 'edge', 'frequency', 'statistical', 'keypoint'])
    new_fused_feature = new_feature_fusion.extract_features()
    print(f"新图像特征维度: {new_fused_feature.shape}")

    prediction = detector.detect(new_fused_feature)
    print(prediction)
