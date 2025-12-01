import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import gaussian_kde

np.random.seed(42)
n = 300

# Uniform data
uniform_x = np.random.uniform(0, 1, n)
uniform_y = np.random.uniform(0, 1, n)
uniform = np.vstack([uniform_x, uniform_y]).T

# Non-uniform data (clustered)
cluster1 = np.random.normal(loc=[0.25, 0.25], scale=0.05, size=(100, 2))
cluster2 = np.random.normal(loc=[0.75, 0.75], scale=0.05, size=(100, 2))
noise = np.random.uniform(0, 1, (100, 2))
nonuniform = np.vstack([cluster1, cluster2, noise])

datasets = {"Uniform": uniform, "Non-uniform": nonuniform}

# --- Plotting functions ---
def plot_dbscan(data, title):
    db = DBSCAN(eps=0.1, min_samples=5).fit(data)
    labels = db.labels_
    plt.figure(figsize=(6,6))
    plt.scatter(data[:,0], data[:,1], c=labels)
    plt.title(f"{title} - DBSCAN Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_meanshift(data, title):
    bw = estimate_bandwidth(data, quantile=0.2)
    ms = MeanShift(bandwidth=bw).fit(data)
    labels = ms.labels_
    centers = ms.cluster_centers_
    plt.figure(figsize=(6,6))
    plt.scatter(data[:,0], data[:,1], c=labels)
    plt.scatter(centers[:,0], centers[:,1], marker='x', s=200)
    plt.title(f"{title} - MeanShift Density Peaks")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_lof(data, title):
    lof = LocalOutlierFactor(n_neighbors=20)
    scores = -lof.fit_predict(data)
    plt.figure(figsize=(6,6))
    plt.scatter(data[:,0], data[:,1], c=scores)
    plt.title(f"{title} - LOF Density Score")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_kde(data, title):
    kde = gaussian_kde(data.T)
    xg, yg = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    coords = np.vstack([xg.flatten(), yg.flatten()])
    zg = kde(coords).reshape(100,100)
    plt.figure(figsize=(6,6))
    plt.contourf(xg, yg, zg)
    plt.title(f"{title} - KDE Density Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# --- Generate all plots for both datasets ---
for name, data in datasets.items():
    plot_dbscan(data, name)
    plot_meanshift(data, name)
    plot_lof(data, name)
    plot_kde(data, name)
##########################################################
##########################################################
datasets = {"Uniform": uniform}
for name, data in datasets.items():
    print(f"Processing dataset: {name}")
    lof = LocalOutlierFactor(n_neighbors=10)
    scores = -lof.fit_predict(data)
    plt.figure(figsize=(6,6))
    plt.scatter(data[:,0], data[:,1], c=scores)
    # plt.title(f"{title} - LOF Density Score")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

##########################################################
##########################################################

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def uniformity_lof_score(data, n_neighbors=20):

    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof_scores = -lof.fit_predict(data)  # 1=正常, -1=outlier (但不夠用)

    # # 真正的 LOF value 用 negative_outlier_factor_
    lof_values = -lof.negative_outlier_factor_

    # # 均勻度指標 = LOF 的標準差反比
    lof_std = np.std(lof_values)
    # lof_std = np.std(lof_scores)
    uniformity_score = 1 / (1 + lof_std)

    return uniformity_score, lof_scores


# --------- 測試用資料 ---------

np.random.seed(42)

# 均勻
uniform = np.random.rand(300, 2)

# 不均勻 (clusters)
cluster1 = np.random.normal([0.25,0.25], 0.05, (120,2))
cluster2 = np.random.normal([0.75,0.75], 0.05, (120,2))
noise = np.random.rand(60, 2)
nonuniform = np.vstack([cluster1, cluster2, noise])

# --------- 計算 ---------
score_uniform, lof_u = uniformity_lof_score(uniform)
score_nonuniform, lof_n = uniformity_lof_score(nonuniform)

print("Uniform dataset score:", score_uniform)
print("Non-uniform dataset score:", score_nonuniform)


