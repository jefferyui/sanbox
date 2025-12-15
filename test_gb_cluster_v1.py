import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
transformer = QuantileTransformer(output_distribution='normal', n_quantiles=50, random_state=42)
# 設定繪圖風格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 生成 1D 離群值資料 (Outlier Scenario)
# ==========================================
np.random.seed(42)

# 群體 A: 50 個點，集中在 0
group_a = np.random.normal(0, 1, 50)
# 群體 B: 50 個點，集中在 10
group_b = np.random.normal(10, 1, 50)
# 離群值: 只有 1 個點，在 150 (極端遠)
outlier = np.array([150])

# 合併資料
X = np.concatenate([group_a, group_b, outlier])
X_reshaped = X.reshape(-1, 1)
X_reshaped = transformer.fit_transform(X_reshaped)

# 建立 DataFrame
df = pd.DataFrame({'Value': X})
# 標記真實類別 (0=群A, 1=群B, 2=離群值)
# 我們希望模型能把 A 和 B 分開，而不要被 Outlier 騙走
df['True_Label'] = ['Group A']*50 + ['Group B']*50 + ['Outlier']*1

# ==========================================
# 2. 訓練模型 (都要分兩群)
# ==========================================

# --- A. K-Means (k=2) ---
# K-Means 會被那個 150 的誤差平方嚇死，試圖去包容它
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['KMeans_Label'] = kmeans.fit_predict(X_reshaped)
kmeans_centers = kmeans.cluster_centers_.flatten()

# --- B. SOM (1x2) ---
# SOM 只有兩個神經元。因為 A 和 B 的點很多(各50個)，Outlier 只有1個
# 學習率遞減的情況下，神經元會停留在密度高的地方 (0 和 10)
som = MiniSom(x=1, y=2, input_len=1, sigma=0.5, learning_rate=0.5, random_seed=42)
som.random_weights_init(X_reshaped)
som.train_random(X_reshaped, 5000) # 訓練多次讓密度效應顯現
som_weights = som.get_weights().flatten()

# 根據 SOM center 預測類別
df['SOM_Label'] = [som.winner(x)[1] for x in X_reshaped]

# ==========================================
# 3. 視覺化比較
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 定義畫圖函數
def plot_1d_result(ax, label_col, centers, title, model_name):
    # 畫散佈點 (加一點 y 軸雜訊 jitter 以防重疊)
    y_jitter = np.random.uniform(-0.1, 0.1, len(df))
    sns.scatterplot(
        data=df, x='Value', y=y_jitter, hue=label_col, 
        palette='Set1', ax=ax, s=60, alpha=0.8
    )
    
    # 畫出中心點位置
    for c in centers:
        ax.axvline(c, color='black', linestyle='--', linewidth=2, label='Center')
        # 標示中心點數值
        ax.text(c, 0.2, f'{model_name}\nCenter:{c:.1f}', ha='center', fontweight='bold')

    # 畫出 Outlier 的箭頭標示
    ax.annotate('Outlier (150)', xy=(150, 0), xytext=(120, 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks([]) # 隱藏 y 軸
    ax.legend(loc='upper right', title='Predicted Cluster')

# --- 繪製 K-Means ---
# 預期：中心點會被拉走，導致 Group A 和 Group B 被合併
plot_1d_result(axes[0], 'KMeans_Label', kmeans_centers, 
               'K-Means (失敗)：被離群值拉走，導致 A/B 兩群合併', 'KMeans')

# --- 繪製 SOM ---
# 預期：中心點穩穩地抓在 0 和 10，無視 Outlier
plot_1d_result(axes[1], 'SOM_Label', som_weights, 
               'SOM (成功)：基於密度競爭，鎖定主要群體 (A/B)', 'SOM')

plt.xlabel('Value', fontsize=12)
# plt.tight_layout()
plt.show()

# 顯示中心點數據
print(f"K-Means Centers: {kmeans_centers}")
print(f"SOM Weights:     {som_weights}")
