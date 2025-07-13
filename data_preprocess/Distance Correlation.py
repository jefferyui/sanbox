import numpy as np
import dcor
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# 模擬數據：10 個特徵，100 條數據點
np.random.seed(42)
X = np.random.rand(100, 10)

# 計算特徵之間的 Distance Correlation
num_features = X.shape[1]
correlation_matrix = np.zeros((num_features, num_features))

for i in range(num_features):
    for j in range(num_features):
        correlation_matrix[i, j] = 1 - dcor.distance_correlation(X[:, i], X[:, j])  # 1 - correlation 作為距離

# 將矩陣轉成距離格式（展平）
distance_vector = correlation_matrix[np.triu_indices(num_features, k=1)]

# 使用層次聚類
linkage_matrix = linkage(distance_vector, method='ward')

# 畫樹狀圖
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=[f"Feat_{i}" for i in range(num_features)])
plt.title('Feature Clustering Based on Distance Correlation')
plt.show()

# 聚類（指定 3 組）
clusters = fcluster(linkage_matrix, 3, criterion='maxclust')
print("特徵聚類結果：", clusters)
