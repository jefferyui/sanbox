import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, poisson

# =====================================================
# 1. 生成示範 Dataset（後期點數變多）
# =====================================================
np.random.seed(42)

# 前期：點數少
t_early = np.random.uniform(0, 48, size=80)

# 後期：點數多（模擬異常前活躍）
t_late = np.random.uniform(48, 72, size=200)

time_hours = np.concatenate([t_early, t_late])
value = np.random.normal(loc=10, scale=1, size=len(time_hours))

df = pd.DataFrame({
    "time": time_hours,
    "value": value
}).sort_values("time").reset_index(drop=True)

# =====================================================
# 用 time bin 計算歷史 expected_24h（SPC 版）
# =====================================================

# 最近時間
latest_time = df["time"].max()
recent_start = latest_time - 24

# 最近 24 小時實際點數
recent_count = df[df["time"] >= recent_start].shape[0]

# -------- 歷史資料（排除最近 24h）--------
historical_df = df[df["time"] < recent_start].copy()

# -------- 切成「每 24 小時一個 bin」--------
# 注意：這裡 time 單位是「小時」
historical_df["time_bin_24h"] = (
    (historical_df["time"] - historical_df["time"].min()) // 24
)

# 每個 24h bin 的點數
hist_24h_counts = (
    historical_df
    .groupby("time_bin_24h")
    .size()
)

# 歷史期望值（24h）
# expected_24h = hist_24h_counts.mean()
expected_24h = hist_24h_counts.median()
# -------- 百分比變化 --------
pct_change = (recent_count - expected_24h) / expected_24h * 100

# -------- 門檻設定 --------
UP_THRESHOLD = 30
DOWN_THRESHOLD = -30

print("\n=== 最近 24 小時點數（Time-bin 百分比判斷） ===")
print(f"最近 24h 點數   = {recent_count}")
print(f"歷史 24h 平均   = {expected_24h:.1f}")
print(f"變化百分比     = {pct_change:.1f}%")

if pct_change >= UP_THRESHOLD:
    print("最近 24 小時【點數明顯變多】")
elif pct_change <= DOWN_THRESHOLD:
    print("最近 24 小時【點數明顯變少】")
else:
    print("最近 24 小時點數在正常範圍")


# =====================================================
# 5. 視覺化：最近 24 小時 vs 歷史
# =====================================================
plt.figure()
plt.hist(
    historical_df["time"],
    bins=20,
    alpha=0.7,
    label="Historical"
)
plt.hist(
    df[df["time"] >= recent_start]["time"],
    bins=10,
    alpha=0.7,
    label="Last 24h"
)
plt.xlabel("Time (hours)")
plt.ylabel("Count")
plt.title("Historical vs Last 24 Hours Point Density")
plt.legend()
plt.show()




########################## ########################## ########################## ########################## 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, poisson

# =====================================================
# 1. 生成示範 Dataset（後期點數變多）
# =====================================================
np.random.seed(42)

# 前期：點數少
t_early = np.random.uniform(0, 48, size=80)

# 後期：點數多（模擬異常前活躍）
t_late = np.random.uniform(48, 72, size=200)

time_hours = np.concatenate([t_early, t_late])
value = np.random.normal(loc=10, scale=1, size=len(time_hours))

df = pd.DataFrame({
    "time": time_hours,
    "value": value
}).sort_values("time").reset_index(drop=True)

# =====================================================
# 2. 方法一：時間 bin 點數趨勢（整體）
# =====================================================
n_bins = 18
df["time_bin"] = pd.cut(df["time"], bins=n_bins)

bin_counts = (
    df.groupby("time_bin")
      .size()
      .reset_index(name="count")
)

bin_counts["bin_index"] = np.arange(len(bin_counts))

# --- 趨勢指標（Spearman） ---
rho, p_trend = spearmanr(
    bin_counts["bin_index"],
    bin_counts["count"]
)

print("=== 整體點數趨勢 ===")
print(f"Spearman rho = {rho:.3f}")
print(f"p-value      = {p_trend:.4f}")

# =====================================================
# 3. 畫圖：SPC 風格點數趨勢
# =====================================================
plt.figure()
plt.plot(
    bin_counts["bin_index"],
    bin_counts["count"],
    marker="o"
)
plt.xlabel("Time bin index")
plt.ylabel("Point count")
plt.title("SPC: Point Count Trend Over Time")
plt.show()


# =====================================================
# 百分比判斷：最近 24 小時是否點數異常
# =====================================================
# # 定義時間
latest_time = df["time"].max()
recent_start = latest_time - 24
# 最近 24 小時實際點數
recent_count = df[df["time"] >= recent_start].shape[0]

# 歷史基準（排除最近 24h）
historical_df = df[df["time"] < recent_start]

total_hist_hours = historical_df["time"].max() - historical_df["time"].min()
expected_24h = len(historical_df) / total_hist_hours * 24

# ---- 百分比變化 ----
pct_change = (recent_count - expected_24h) / expected_24h * 100

# ---- 門檻設定（可依製程調）----
UP_THRESHOLD = 30    # +30%
DOWN_THRESHOLD = -30 # -30%

print("\n=== 最近 24 小時點數（百分比判斷） ===")
print(f"最近 24h 點數   = {recent_count}")
print(f"歷史 24h 平均   = {expected_24h:.1f}")
print(f"變化百分比     = {pct_change:.1f}%")

if pct_change >= UP_THRESHOLD:
    print("最近 24 小時【點數明顯變多】")
elif pct_change <= DOWN_THRESHOLD:
    print("最近 24 小時【點數明顯變少】")
else:
    print("最近 24 小時點數在正常範圍")


# =====================================================
# 5. 視覺化：最近 24 小時 vs 歷史
# =====================================================
plt.figure()
plt.hist(
    historical_df["time"],
    bins=20,
    alpha=0.7,
    label="Historical"
)
plt.hist(
    df[df["time"] >= recent_start]["time"],
    bins=10,
    alpha=0.7,
    label="Last 24h"
)
plt.xlabel("Time (hours)")
plt.ylabel("Count")
plt.title("Historical vs Last 24 Hours Point Density")
plt.legend()
plt.show()
