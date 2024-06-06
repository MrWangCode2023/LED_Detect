import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def optimal_dbscan(bgr_values):
    """
    Perform DBSCAN clustering on BGR values and automatically determine the optimal parameters.

    Parameters:
    - bgr_values: A numpy array of shape (n_samples, 3) containing BGR pixel values.

    Returns:
    - best_eps: Optimal eps value.
    - best_min_samples: Optimal min_samples value.
    - best_centers: Cluster centers (BGR values).
    - best_labels: Cluster labels for each pixel.
    - best_silhouette_score: Best silhouette score achieved.
    """
    best_eps = 0
    best_min_samples = 0
    best_centers = None
    best_labels = None
    best_silhouette_score = -1

    # Define ranges for eps and min_samples
    eps_values = np.linspace(0.1, 10.0, 100)  # Adjust the range and step
    min_samples_values = range(2, 10)  # Adjust the range

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(bgr_values)

            # Ensure we have more than one cluster and at least one core sample
            unique_labels = set(labels)
            if len(unique_labels) > 1 and -1 in unique_labels:
                silhouette_avg = silhouette_score(bgr_values, labels)
                if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg
                    best_eps = eps
                    best_min_samples = min_samples
                    best_labels = labels
                    # Calculate cluster centers
                    cluster_centers = []
                    for label in unique_labels:
                        if label != -1:  # Exclude noise points
                            cluster_points = bgr_values[labels == label]
                            cluster_center = np.mean(cluster_points, axis=0)
                            cluster_centers.append(cluster_center)
                    best_centers = np.array(cluster_centers)

    return best_eps, best_min_samples, best_centers, best_labels, best_silhouette_score


# 示例像素值数据
num_points = 10000
bgr_points = np.random.randint(0, 256, size=(num_points, 3), dtype=np.uint8)

# 自动确定最佳DBSCAN参数并进行聚类
best_eps, best_min_samples, centers, labels, silhouette_avg = optimal_dbscan(bgr_points)

print(f"Optimal eps: {best_eps}")
print(f"Optimal min_samples: {best_min_samples}")
if centers is not None:
    print(f"Cluster centers (BGR values): {centers}")
else:
    print("No cluster centers found.")
print(f"Cluster labels: {labels}")
print(f"Best silhouette score: {silhouette_avg}")
