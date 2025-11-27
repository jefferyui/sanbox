import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score
# 注意：這裡引入的是 MultKAN，這是 KAN 2.0 的核心
from kan import MultKAN
import matplotlib.pyplot as plt

# ==========================================
# 1. 生成資料 (具有交互作用的資料)
# ==========================================
np.random.seed(42)
n_samples = 1000

data = {
    'Material': np.random.choice(['Wood', 'Metal', 'Plastic'], n_samples),
    'Quality': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Size': np.random.uniform(1, 10, n_samples)
}
df = pd.DataFrame(data)

# 真實函數：包含明顯的 "乘法交互作用"
# 價格 = (材質基價 * 品質加成) + Size^2
# 這種 (A * B) 結構正是 MultKAN 最擅長的
def true_price_function(row):
    base = 50 if row['Material'] == 'Wood' else (100 if row['Material'] == 'Metal' else 20)
    
    mult = 1.0
    if row['Quality'] == 'Low': mult = 0.8
    elif row['Quality'] == 'High': mult = 1.5
    
    # 加上隨機噪音
    return (base * mult) + (row['Size'] ** 2) + np.random.normal(0, 5)

df['Price'] = df.apply(true_price_function, axis=1)

# ==========================================
# 2. 特徵工程 (One-Hot + Scaling)
# ==========================================
X = df[['Material', 'Quality', 'Size']]
y = df['Price'].values

# A. One-Hot Encoding
# 我們希望保留所有類別資訊供 MultKAN 進行組合
encoder = OneHotEncoder(sparse_output=False)
X_cat_encoded = encoder.fit_transform(X[['Material', 'Quality']])

# B. 合併連續變數
X_final = np.hstack([X_cat_encoded, X[['Size']].values])

# C. 全局正規化到 [-1, 1]
# 即使是 One-Hot (0/1)，縮放到 -1/1 也能讓它利用到 B-Spline 的整個定義域，效果通常更好
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_final)

# D. 轉 Tensor
train_inputs, test_inputs, train_label, test_label = train_test_split(
    torch.tensor(X_scaled, dtype=torch.float32),
    torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    test_size=0.2,
    random_state=42
)

dataset = {
    'train_input': train_inputs,
    'train_label': train_label,
    'test_input': test_inputs,
    'test_label': test_label
}

# ==========================================
# 3. 建立 KAN 2.0 (MultKAN) 模型
# ==========================================

input_dim = X_final.shape[1]
output_dim = 1

# MultKAN 初始化
# width: 結構 [輸入, 隱藏, 輸出]
# grid: 網格大小
# k: Spline 階數 (3 = Cubic)
# mult_arity: 這是 2.0 的關鍵參數！
#   mult_arity=2 表示允許兩個特徵相乘 (例如 Material * Quality)。
#   如果資料非常複雜，可以設為 3，但 2 對大多數回歸足夠且更穩定。
model = MultKAN(width=[input_dim, 5, output_dim], 
                grid=5, 
                k=3, 
                mult_arity=2) 

print(f"--- KAN 2.0 模型結構 ---")
print(f"輸入維度: {input_dim}")
print(f"使用 MultKAN，允許特徵交互作用 (Multiplication)")

# ==========================================
# 4. 訓練 (使用 LBFGS)
# ==========================================
print("\n--- 開始訓練 ---")
# KAN 2.0 訓練通常需要稍微多一點的 steps 來調整乘法節點
results = model.fit(dataset, opt="LBFGS", steps=25, lamb=0.01)

# ==========================================
# 5. 評估結果
# ==========================================
pred_train = model(dataset['train_input']).detach().numpy()
pred_test = model(dataset['test_input']).detach().numpy()
y_train = dataset['train_label'].detach().numpy()
y_test = dataset['test_label'].detach().numpy()

r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)

print("\n" + "="*30)
print(f"訓練集 R-squared: {r2_train:.4f}")
print(f"測試集 R-squared: {r2_test:.4f}")
print("="*30)

# 如果想看交互作用圖 (KAN 2.0 的特色)
model.plot()
plt.show()

# 這裡不需要建立 figure，pandas 會自動處理
df.boxplot(column='Price', by=['Material', 'Quality'], figsize=(12, 6))

plt.title('材質與品質分組盒鬚圖')
plt.suptitle('') # 移除 pandas 自動產生的預設標題以保持整潔
plt.xlabel('材質 - 品質')
plt.ylabel('價格')
plt.xticks(rotation=45) # 如果標籤太擠，可以旋轉角度
plt.show()


from sympy import *
import sympy

# ==========================================
# 步驟 1: 剪枝 (Pruning) - 已修正參數
# ==========================================
print("--- 步驟 1: 剪枝 (Pruning) ---")
# 使用 node_th 代替 threshold
model.prune(node_th=0.01) 

# ==========================================
# 步驟 2: 符號化 (Symbolic Regression)
# ==========================================
print("--- 步驟 2: 符號化 (Symbolic Regression) ---")
# 限制函數庫，讓公式不要太奇怪
lib = ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'sin', 'abs']

# 這步會需要一點時間計算
model.auto_symbolic(lib=lib)

# ==========================================
# 步驟 3: 印出公式
# ==========================================
print("--- 步驟 3: 印出公式 ---")

# 檢查是否有公式生成
if len(model.symbolic_formula()) > 0:
    # 取得第一個輸出維度的公式
    formula = model.symbolic_formula()[0][0]
    
    print("\n=== KAN 原始公式 (x_1, x_2...) ===")
    sympy.pprint(formula)

    # ---------------------------------------------------
    # 替換成真實特徵名稱 (Readable)
    # ---------------------------------------------------
    # 注意：這裡需要根據您 X_final 的實際特徵名稱來對應
    # 如果您前面使用了 OneHotEncoder，feature_names_encoded 變數應該還在
    # 這裡我們重新構建一次以防萬一：
    
    # 取得 One-Hot 特徵名
    if 'encoder' in locals():
        cat_names = encoder.get_feature_names_out(['Material', 'Quality'])
        # 加上連續變數 'Size'
        all_names = list(cat_names) + ['Size']
    else:
        # 如果變數遺失，這裡手動定義範例 (請依照您的數據順序修改)
        all_names = ['Mat_Metal', 'Mat_Plastic', 'Mat_Wood', 'Qual_High', 'Qual_Low', 'Qual_Med', 'Size']

    # 建立替換字典
    subs_dict = {}
    for i, name in enumerate(all_names):
        # SymPy 的符號通常是 x_1, x_2 (index 從 1 開始)
        kan_var = sympy.symbols(f'x_{i+1}')
        # 清理特徵名稱中的特殊字符，避免 SymPy 報錯
        clean_name = name.replace(' ', '_').replace('-', '_')
        real_var = sympy.symbols(clean_name)
        subs_dict[kan_var] = real_var  # 對應到上一行的 real_var
