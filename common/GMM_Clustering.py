import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
import random


def optimal_gmm(bgr_values, n_jobs=-1):
    """
    Perform GMM clustering on BGR values and automatically determine the optimal number of components.

    Parameters:
    - bgr_values: A numpy array of shape (n_samples, 3) containing BGR pixel values.
    - n_jobs: Number of parallel jobs to run. -1 means using all processors.

    Returns:
    - best_n_components: Optimal number of components.
    - best_labels: Cluster labels for each pixel.
    - best_centers: Cluster centers (BGR values).
    - best_silhouette_score: Best silhouette score achieved.
    """
    best_n_components = 0
    best_labels = None
    best_centers = None
    best_silhouette_score = -1

    # Define range for number of components
    n_components_range = range(2, min(10, len(bgr_values)))  # Ensure n_components <= n_samples

    def fit_gmm(n_components):
        gmm = GaussianMixture(n_components=n_components, max_iter=100, random_state=42)
        gmm.fit(bgr_values)
        labels = gmm.predict(bgr_values)
        centers = gmm.means_
        silhouette_avg = silhouette_score(bgr_values, labels)
        return n_components, labels, centers, silhouette_avg

    results = Parallel(n_jobs=n_jobs)(delayed(fit_gmm)(n_components) for n_components in n_components_range)

    for n_components, labels, centers, silhouette_avg in results:
        # print(f"n_components: {n_components}, silhouette_avg: {silhouette_avg}")
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_components = n_components
            best_labels = labels
            best_centers = centers
            num_centers = len(best_centers)

    # return best_n_components, best_silhouette_score, best_labels, best_centers
    return best_centers, num_centers


# 示例像素值数据
num_points = 100000
bgr_points = np.random.randint(0, 256, size=(num_points, 3), dtype=np.uint8)

# 下采样数据，例如将数据量减少到原来的10%
sampled_bgr_points = bgr_points[random.sample(range(num_points), num_points // 15)]

# 自动确定最佳GMM成分数并进行聚类
best_centers, num_centers = optimal_gmm(sampled_bgr_points)

# print(f"Optimal number of components: {best_n_components}")
# print(f"Best silhouette score: {best_silhouette_score}")
# print(f"Cluster labels: {best_labels}")
# print(f"Cluster centers (BGR values): {best_centers}")
print("聚类中心：\n", best_centers)
print("类别个数：", num_centers)