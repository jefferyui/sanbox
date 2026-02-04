import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1️⃣ 生成時間序列資料
# =========================
np.random.seed(42)
N = 300_000   # 可改成 1_000_000

time_index = pd.date_range(
    start="2024-01-01",
    periods=N,
    freq="S"
)

trend = np.linspace(0, 5, N)
noise = np.random.randn(N) * 0.8
y = np.sin(np.linspace(0, 200, N)) + trend * 0.05 + noise

# =========================
# 2️⃣ 超高速 smooth (推薦🔥 convolution)
# =========================
def fast_smooth(y, window=500):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

y_smooth = fast_smooth(y, window=500)

# =========================
# 3️⃣ 超高速畫圖 (line + scatter + smooth)
# =========================
plt.figure(figsize=(12, 5))

# ✅ 原始 line (downsample)
# step = 10
# plt.plot(time_index[::step], y[::step], linewidth=0.5, alpha=0.4, label="raw (downsample)")
step = 1
plt.plot(time_index[::step], y[::step], linewidth=0.5, alpha=0.4, label="raw (downsample)")
# ✅ scatter (full data + rasterized)
# plt.scatter(time_index, y, s=1, marker='.', alpha=0.25,
#             edgecolors='none', rasterized=True)
plt.plot(time_index, y, '.', markersize=1, alpha=0.4)
# ✅ smooth line (downsample)
plt.plot(time_index[::step], y_smooth[::step], color="red", linewidth=2, label="smooth")

plt.title("Time Series + Fast Scatter + Fast Smooth Line")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()


##############################3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1️⃣ 生成時間序列資料 (100萬筆)
# =========================
np.random.seed(42)
N = 300_000   # 建議先用 30萬，避免太慢

time_index = pd.date_range(
    start="2024-01-01",
    periods=N,
    freq="S"   # 每秒一筆
)

# 模擬時間序列訊號
trend = np.linspace(0, 10, N)
noise = np.random.randn(N) * 0.8
y = np.sin(np.linspace(0, 200, N)) + trend * 0.05 + noise

# =========================
# 2️⃣ Matplotlib 加速 scatter + line plot
# =========================
plt.figure(figsize=(12, 5))

# ✅ 超快 line plot（建議先畫 line）
plt.plot(
    time_index,
    y,
    linewidth=0.5,     # 線細一點
    alpha=0.6
)

# plt.plot(time_index, y, '.', markersize=1, alpha=0.4)
# plt.title("Plot Instead of Scatter (Very Fast)")
# plt.show()
# ✅ 加速 scatter
plt.scatter(
    time_index,
    y,
    s=1,                # 小點
    marker='.',         # 最快 marker
    alpha=0.4,
    edgecolors='none',
    rasterized=True     # 超重要🔥
)

plt.title("Time Series: Line + Fast Scatter")
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
