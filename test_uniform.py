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

