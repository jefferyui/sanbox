import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde

# -----------------------
# 1. Grid CV Uniformity
# -----------------------
def grid_uniformity_score(data, grid=10):
    x, y = data[:,0], data[:,1]
    H, _, _ = np.histogram2d(x, y, bins=grid)
    counts = H.flatten()
    mean = counts.mean()
    std = counts.std()
    if mean == 0:
        return 0
    cv = std / mean
    return 1 / (1 + cv)


# -----------------------
# 2. KNN Uniformity
# -----------------------
def knn_uniformity_score(data, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)
    avg_dist = distances.mean(axis=1)
    return 1 / (1 + np.std(avg_dist))


# -----------------------
# 3. KDE Uniformity
# -----------------------
def kde_uniformity_score(data):
    kde = gaussian_kde(data.T)
    xs = np.random.rand(500)
    ys = np.random.rand(500)
    samples = np.vstack([xs, ys])
    densities = kde(samples)
    return 1 / (1 + np.std(densities))


# -----------------------
# 建立兩組資料（均勻 vs 不均勻）
# -----------------------

np.random.seed(42)

uniform = np.random.rand(300, 2)

cluster1 = np.random.normal([0.3, 0.3], 0.05, (100,2))
cluster2 = np.random.normal([0.7, 0.7], 0.05, (100,2))
noise = np.random.rand(100, 2)
nonuniform = np.vstack([cluster1, cluster2, noise])


# -----------------------
# 計算分數
# -----------------------

datasets = {"Uniform": uniform, "Non-uniform": nonuniform}

for name, data in datasets.items():

    g = grid_uniformity_score(data)
    k = knn_uniformity_score(data)
    d = kde_uniformity_score(data)

    print(f"\n=== {name} Dataset ===")
    print(f"Grid CV Uniformity Score: {g:.4f}")
    print(f"KNN Uniformity Score:     {k:.4f}")
    print(f"KDE Uniformity Score:     {d:.4f}")

    # 也可以畫圖
    plt.figure(figsize=(6,6))
    plt.scatter(data[:,0], data[:,1])
    plt.title(f"{name} Scatter")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
######################################################################



# --- Generate datasets ---
import pandas as pd
datasets = {}

np.random.seed(0)

datasets["uniform"] = np.random.rand(500, 2)

cluster1 = np.random.randn(200, 2) * 0.05 + np.array([0.3, 0.7])
cluster2 = np.random.randn(200, 2) * 0.05 + np.array([0.7, 0.3])
datasets["clustered"] = np.vstack([cluster1, cluster2])

theta = np.random.rand(500) * 2 * np.pi
r = 0.1 + 0.02*np.random.randn(500)
datasets["ring"] = np.c_[0.5 + r*np.cos(theta), 0.5 + r*np.sin(theta)]

x = np.random.beta(0.3, 0.3, 500)
y = np.random.rand(500)
datasets["x_bias"] = np.c_[x, y]

corners = np.random.choice(4, 500)
coords = []
corner_centers = [(0.1,0.1),(0.9,0.1),(0.1,0.9),(0.9,0.9)]
for c in corners:
    coords.append(np.random.randn(2)*0.03 + corner_centers[c])
datasets["corner_heavy"] = np.array(coords)

x = np.repeat(np.linspace(0.1,0.9,5), 100)
y = np.random.rand(500)
datasets["vertical_bands"] = np.c_[x + 0.02*np.random.randn(500), y]

y = np.repeat(np.linspace(0.1,0.9,5), 100)
x = np.random.rand(500)
datasets["horizontal_bands"] = np.c_[x, y + 0.02*np.random.randn(500)]

datasets["gaussian_blob"] = np.random.randn(500,2)*0.1 + np.array([0.5,0.5])

results=[]
for name, data in datasets.items():

    g = grid_uniformity_score(data)
    k = knn_uniformity_score(data)
    d = kde_uniformity_score(data)

    print(f"\n=== {name} Dataset ===")
    print(f"Grid CV Uniformity Score: {g:.4f}")
    print(f"KNN Uniformity Score:     {k:.4f}")
    print(f"KDE Uniformity Score:     {d:.4f}")

    # 也可以畫圖
    plt.figure(figsize=(6,6))
    plt.scatter(data[:,0], data[:,1])
    plt.title(f"{name} Scatter")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    final_score = (g + k + d) / 3

    results.append([name, g, k, d, final_score])

df = pd.DataFrame(results, columns=["dataset", "grid_score", "knn_score", "kde_score", "final_uniformity"])
df


✅ 方法 1：Grid Density Variance（最推薦，最穩定）

把整個 XY 平面分成 10×10 格子（可調），
每個格子應該都有類似數量的點 → 才代表均勻分佈。

衡量指標：

Grid Uniformity Score = 1 / (1 + CV(grid_counts))
CV = 點數的變異係數 = std / mean

🟢 score 越接近 1 → 越均勻

🔴 分布越偏、越有 cluster → score 越低

✅ 方法 2：KNN 距離變異度（Local Density Variation）

均勻分佈時：

每個點到鄰近 10 個點的平均距離都差不多

不均勻會出現：cluster（距離短）＋稀疏區（距離長）

衡量指標：

KNN Uniformity Score = 1 / (1 + std(avg_distances))

越接近 1 → XY 越均勻

越接近 0 → 有 cluster 或空洞

✅ 方法 3：KDE 密度平滑度（Density Smoothness）

對 XY 做 Kernel Density Estimation，均勻時：

密度曲面非常平滑（變化小）

衡量指標：

KDE Uniformity Score = 1 / (1 + std(density_values))

越平滑 → 越均勻

出現峰值（cluster）→ 分數下降
