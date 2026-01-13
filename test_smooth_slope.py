import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==============================
# 1. 模擬 SPC 資料（含後期上升）
# ==============================
np.random.seed(42)

n_days = 60
dates = pd.date_range("2025-01-01", periods=n_days)

# 前期穩定 + 後期輕微上升
values = np.concatenate([
    np.random.normal(loc=100, scale=2, size=50),
    np.random.normal(loc=105, scale=2, size=10)
])

df = pd.DataFrame({
    "date": dates,
    "value": values
})

# ==============================
# 2. SPC 控制界線（I-Chart）
# ==============================
mean = df["value"].mean()
std = df["value"].std()

df["UCL"] = mean + 3 * std
df["LCL"] = mean - 3 * std
df["CL"] = mean

# ==============================
# 3. Smooth line（rolling mean）
# ==============================
window = 7
df["smooth"] = df["value"].rolling(window=window, min_periods=1).mean()

# ==============================
# 4. 切分資料：最近10天 vs 之前
# ==============================
recent_days = 10

df_recent = df.iloc[-recent_days:]
df_past = df.iloc[:-recent_days]

# ==============================
# 5. 計算 smooth line 斜率
# ==============================
def calc_slope(y):
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0]

slope_recent = calc_slope(df_recent["smooth"].values)
slope_past = calc_slope(df_past["smooth"].values)

# ==============================
# 6. 比較 smooth line 範圍
# ==============================
recent_min, recent_max = df_recent["smooth"].min(), df_recent["smooth"].max()
past_min, past_max = df_past["smooth"].min(), df_past["smooth"].max()

range_exceed = (recent_min < past_min) or (recent_max > past_max)

# ==============================
# 7. 視覺化 SPC + smooth
# ==============================
plt.figure(figsize=(12, 6))

plt.plot(df["date"], df["value"], marker="o", label="Raw data", alpha=0.6)
plt.plot(df["date"], df["smooth"], linewidth=3, label="Smooth line")

plt.plot(df["date"], df["UCL"], linestyle="--", color="red", label="UCL")
plt.plot(df["date"], df["CL"], linestyle="--", color="black", label="CL")
plt.plot(df["date"], df["LCL"], linestyle="--", color="red", label="LCL")

# 標示最近10天
plt.axvspan(df_recent["date"].iloc[0], df_recent["date"].iloc[-1],
            color="orange", alpha=0.2, label="Recent 10 days")

plt.title("SPC Chart with Smooth Line")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 8. 結果判讀輸出
# ==============================
print("=== Smooth Line 趨勢分析 ===")
print(f"過去 smooth 斜率     : {slope_past:.4f}")
print(f"最近10天 smooth 斜率 : {slope_recent:.4f}")

if slope_recent > slope_past * 1.5:
    print("⚠️ 趨勢警告：最近10天上升斜率明顯增加")
else:
    print("✅ 趨勢穩定")

print("\n=== Smooth Line 範圍分析 ===")
print(f"歷史範圍  : [{past_min:.2f}, {past_max:.2f}]")
print(f"最近10天 : [{recent_min:.2f}, {recent_max:.2f}]")

if range_exceed:
    print("⚠️ 水準警告：最近10天 smooth 超出歷史範圍")
else:
    print("✅ 水準仍在歷史範圍內")



###########################################################################3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==============================
# 1. 模擬資料（後段有上升）
# ==============================
np.random.seed(42)

n = 60
dates = pd.date_range("2025-01-01", periods=n)

values = np.concatenate([
    np.random.normal(100, 2, 45),
    np.random.normal(106, 2, 15)
])

df = pd.DataFrame({
    "date": dates,
    "value": values
})

# ==============================
# 2. Smooth line（rolling mean）
# ==============================
window = 7
df["smooth"] = df["value"].rolling(window, min_periods=1).mean()

# ==============================
# 3. Uniform grid 切 5 段
# ==============================
n_grid = 5
df["grid"] = pd.cut(
    np.arange(len(df)),
    bins=n_grid,
    labels=False
)

# ==============================
# 4. 每個 grid 計算 slope 與 range
# ==============================
def calc_slope(y):
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0]

grid_stats = []

for g in range(n_grid):
    sub = df[df["grid"] == g]
    slope = calc_slope(sub["smooth"].values)
    min_v = sub["smooth"].min()
    max_v = sub["smooth"].max()

    grid_stats.append({
        "grid": g,
        "slope": slope,
        "min": min_v,
        "max": max_v
    })

stats_df = pd.DataFrame(grid_stats)

# ==============================
# 5. 最近 grid vs 過去 grids
# ==============================
recent = stats_df.iloc[-1]
past = stats_df.iloc[:-1]

# (A) slope 比較
past_slope_mean = past["slope"].mean()
slope_ratio = recent["slope"] / (past_slope_mean + 1e-6)

# (B) range 比較
past_min = past["min"].min()
past_max = past["max"].max()

range_exceed = (
    recent["min"] < past_min or
    recent["max"] > past_max
)

# ==============================
# 6. 視覺化
# ==============================
plt.figure(figsize=(12, 6))

plt.plot(df["date"], df["value"], marker="o", alpha=0.5, label="Raw")
plt.plot(df["date"], df["smooth"], linewidth=3, label="Smooth")

for g in range(n_grid):
    grid_start = df[df["grid"] == g]["date"].iloc[0]
    grid_end = df[df["grid"] == g]["date"].iloc[-1]
    plt.axvspan(grid_start, grid_end, alpha=0.08)

plt.axvspan(
    df[df["grid"] == n_grid - 1]["date"].iloc[0],
    df[df["grid"] == n_grid - 1]["date"].iloc[-1],
    color="orange",
    alpha=0.25,
    label="Recent grid"
)

plt.title("SPC with Smooth Line (Uniform 5 Grids)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 7. 判斷輸出
# ==============================
print("=== Grid-based Smooth Analysis ===")
print(stats_df, "\n")

print(f"過去 grids 平均 slope : {past_slope_mean:.4f}")
print(f"最近 grid slope      : {recent['slope']:.4f}")
print(f"Slope ratio          : {slope_ratio:.2f}")

if slope_ratio > 1.5:
    print("⚠️ 趨勢異常：最近 grid 上升速度明顯增加")
else:
    print("✅ 趨勢穩定")

print("\n歷史 smooth range :", f"[{past_min:.2f}, {past_max:.2f}]")
print("最近 grid range   :", f"[{recent['min']:.2f}, {recent['max']:.2f}]")

if range_exceed:
    print("⚠️ 水準異常：最近 grid 超出歷史範圍")
else:
    print("✅ 水準仍在歷史範圍內")

