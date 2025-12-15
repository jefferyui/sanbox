import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def gen_uniform(n=1000):
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    return np.c_[x, y]

def gen_horizontal_bands(n=1000, bands=5, noise=0.02):
    x = np.random.uniform(0, 1, n)
    centers = np.linspace(0.1, 0.9, bands)
    y = np.random.choice(centers, n) + np.random.normal(0, noise, n)
    return np.c_[x, y]

def gen_vertical_bands(n=1000, bands=5, noise=0.02):
    y = np.random.uniform(0, 1, n)
    centers = np.linspace(0.1, 0.9, bands)
    x = np.random.choice(centers, n) + np.random.normal(0, noise, n)
    return np.c_[x, y]

def gen_clusters(n=1000):
    centers = [(0.3,0.3), (0.7,0.7), (0.3,0.7)]
    pts = []
    for cx, cy in centers:
        pts.append(np.random.normal([cx, cy], 0.05, (n//3, 2)))
    return np.vstack(pts)

def gen_mixed(n=1000):
    d1 = gen_uniform(n//2)
    d2 = gen_horizontal_bands(n//2, bands=4)
    return np.vstack([d1, d2])

def estimate_modes_1d(values, max_components=6):
    values = values.reshape(-1, 1)
    best_bic = np.inf
    best_k = 1

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(k, random_state=0)
        gmm.fit(values)
        bic = gmm.bic(values)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    return best_k

def auto_bins_cluster_aware(values,
                            min_bins=5,
                            max_bins=50,
                            bins_per_cluster=4):
    n = len(values)
    k = estimate_modes_1d(values)

    bins_cluster = k * bins_per_cluster
    # bins_cluster = k #* bins_per_cluster
    bins_n = int(np.sqrt(n))

    bins = min(bins_cluster, bins_n)
    bins = int(np.clip(bins, min_bins, max_bins))

    return bins, k

def axis_uniformity_score(values, bins):
    hist, _ = np.histogram(values, bins=bins)
    m = hist.mean()
    if m == 0:
        return 0
    cv = hist.std() / m
    return 1 / (1 + cv)

def grid_uniformity_xy_cluster_aware(data):
    x = data[:, 0]
    y = data[:, 1]

    bins_x, kx = auto_bins_cluster_aware(x)
    bins_y, ky = auto_bins_cluster_aware(y)

    ux = axis_uniformity_score(x, bins_x)
    uy = axis_uniformity_score(y, bins_y)

    bins_xy = int(np.mean([bins_x, bins_y]))
    H, _, _ = np.histogram2d(x, y, bins=bins_xy)
    counts = H.flatten()
    m = counts.mean()
    uxy = 0 if m == 0 else 1 / (1 + counts.std() / m)

    return {
        "uniformity_x": ux,
        "uniformity_y": uy,
        "uniformity_xy": uxy,
        "bins_x": bins_x,
        "bins_y": bins_y,
        "modes_x": kx,
        "modes_y": ky
    }
def spc_structure_decision(scores, th=0.75):
    ux, uy = scores["uniformity_x"], scores["uniformity_y"]
    kx, ky = scores["modes_x"], scores["modes_y"]

    if ux > th and uy > th and kx == 1 and ky == 1:
        return "Uniform"

    if ux > th and uy < th and ky >= 3:
        return "Horizontal bands"

    if ux < th and uy > th and kx >= 3:
        return "Vertical bands"

    if kx >= 2 and ky >= 2:
        return "Clusters / mixture"

    return "Weak / transitional"
datasets = {
    "Uniform": gen_uniform(),
    "Horizontal": gen_horizontal_bands(),
    "Vertical": gen_vertical_bands(),
    "Clusters": gen_clusters(),
    "Mixed": gen_mixed()
}

for name, data in datasets.items():
    scores = grid_uniformity_xy_cluster_aware(data)
    decision = spc_structure_decision(scores)

    print(f"\n{name}")
    print("Decision:", decision)
    print(scores)

    plt.figure(figsize=(4,4))
    plt.scatter(data[:,0], data[:,1], s=5)
    plt.title(name)
    plt.show()
