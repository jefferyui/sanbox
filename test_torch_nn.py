import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

# class BostonMLP(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
    
#     def forward(self, x):
#         return self.net(x)


# class BostonMLP(nn.Module):
#     def __init__(self, input_size, output_dim=1, task="regression"):
#         super().__init__()
#         self.task = task
#         self.model = nn.Sequential(
#         nn.Linear(input_size, 64),
#         nn.GELU(),
#         nn.Dropout(0.1),
#         nn.Linear(64, 32),
#         nn.GELU(),
#         nn.Dropout(0.1),
#         nn.Linear(32, output_dim),
#         )
#         if task == "classification":
#             self.final = nn.Sigmoid()
#         else:
#             self.final = nn.Identity()


#     def forward(self, x):
#         x = self.model(x)
#         return self.final(x)

class BostonMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)


# 3. 定義 MLP 模型 (針對回歸優化)
# class BostonMLP(nn.Module):
#     def __init__(self, input_size):
#         super(BostonMLP, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1) # 回歸輸出層不需要 Activation Function
#         )

#     def forward(self, x):
#         return self.network(x)
def train_boston():
    # data = load_boston()
    # X = data.data
    # y = data.target.reshape(-1, 1)
    # 1. 準備數據 (讀取原始來源)
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    X = data
    y = target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scaler_y = StandardScaler() # 對目標值也做標準化通常有助於訓練
    y = scaler_y.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    model = BostonMLP(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(500):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    # Evaluate
    with torch.no_grad():
        pred_test = model(X_test).numpy()
        r2 = r2_score(y_test, pred_test)
        print("Boston Housing R^2:", r2)

train_boston()


#######################
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. 準備數據 (直接從網路讀取)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. 特徵工程 (這是高準確率的關鍵)
# 選擇特徵：船艙等級, 性別, 年齡, 兄弟姊妹數, 父母子女數, 票價, 登船港口
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived'].values

# 定義預處理流程
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# 數值：填補缺失值 (用平均數) -> 標準化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 類別：填補缺失值 (用最頻數) -> One-Hot 編碼
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 處理數據
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 轉為 PyTorch Tensor
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# 3. 定義 MLP 模型 (針對分類優化)
class TitanicNet(nn.Module):
    def __init__(self, input_size):
        super(TitanicNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止過擬合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 輸出一個數值 (Logits)
        )

    def forward(self, x):
        return self.network(x)

# 4. 訓練模型
model = TitanicNet(X_train.shape[1])
criterion = nn.BCEWithLogitsLoss() # 內建 Sigmoid，數值更穩定
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# 5. 評估 (Accuracy)
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    # 將 Logits 轉為機率，大於 0.5 視為存活 (1)
    predicted = (torch.sigmoid(test_outputs) > 0.5).float()
    accuracy = (predicted.eq(y_test_t).sum() / y_test_t.shape[0]).item()

print(f"Titanic Model Accuracy: {accuracy * 100:.2f}%")
print("(Note: Titanic is a classification task, so we use Accuracy instead of R-squared)")

##################
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==========================================
# 1. 產生模擬數據 (二手車數據)
# ==========================================
# 假設數據：品牌, 車型, 產地 -> 價格
data = {
    'Brand': ['Toyota', 'BMW', 'Honda', 'Ferrari', 'Toyota', 'BMW', 'Honda', 'Ferrari'] * 100,
    'Type':  ['Sedan', 'SUV', 'Sedan', 'Sport', 'SUV', 'Sedan', 'SUV', 'Sport'] * 100,
    'Origin':['Japan', 'Germany', 'Japan', 'Italy', 'Japan', 'Germany', 'Japan', 'Italy'] * 100,
    # 價格邏輯 (模擬)：Ferrari > BMW > Toyota/Honda, Sport > SUV > Sedan
    'Price': [20000, 60000, 22000, 250000, 25000, 55000, 28000, 260000] * 100
}
df = pd.DataFrame(data)

# 加入一些隨機雜訊，讓回歸任務真實一點
df['Price'] = df['Price'] + np.random.normal(0, 2000, len(df))

# ==========================================
# 2. 數據預處理 (Label Encoding)
# ==========================================
# Embedding 層需要輸入 "索引 (Index)" (0, 1, 2...)，而不是 One-Hot
categorical_cols = ['Brand', 'Type', 'Origin']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[categorical_cols].values
y = df['Price'].values

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 轉 Tensor
X_train_t = torch.LongTensor(X_train) # 注意：Embedding 輸入必須是 Long (整數)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_test_t = torch.LongTensor(X_test)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# ==========================================
# 3. 定義帶有 Embedding 的 MLP 模型
# ==========================================
class CarPriceNet(nn.Module):
    def __init__(self, embedding_sizes, output_size=1):
        super(CarPriceNet, self).__init__()
        
        # 建立 Embedding 層列表
        # embedding_sizes 格式: [(品牌數, 向量維度), (車型數, 向量維度)...]
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) 
            for num_categories, emb_dim in embedding_sizes
        ])
        
        # 計算全連接層的輸入維度 (所有 Embedding 向量長度的總和)
        self.n_emb = sum(e.embedding_dim for e in self.embeddings)
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.n_emb, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), # 用於加速收斂
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        embeddings = []
        # 對每一列特徵進行 Embedding 轉換
        for i, emb_layer in enumerate(self.embeddings):
            val = x[:, i]             # 取出第 i 列
            emb = emb_layer(val)      # 轉換成向量
            embeddings.append(emb)
        
        # 將所有特徵的向量拼接在一起
        x = torch.cat(embeddings, 1) 
        
        # 進入 MLP
        x = self.mlp_layers(x)
        return x

# 設定 Embedding 大小
# 規則通常是: min(50, (x + 1) // 2)
cat_dims = [len(label_encoders[col].classes_) for col in categorical_cols]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

print(f"Embedding Dimensions: {emb_dims}") 
# 例如: [(4, 2), (3, 2), (3, 2)] -> 代表品牌有4種，我們用2維向量代表它

model = CarPriceNet(emb_dims)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 4. 訓練
# ==========================================
epochs = 1000
loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.0f}")

# ==========================================
# 5. 評估 R-squared
# ==========================================
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t)
    
    # 轉回 numpy
    y_pred_np = y_pred.numpy()
    y_test_np = y_test_t.numpy()
    
    r2 = r2_score(y_test_np, y_pred_np)

print(f"\nFinal R-squared: {r2:.4f}")

#################
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 產生模擬數據 (跟剛才一樣)
# ==========================================
data = {
    'Brand': ['Toyota', 'BMW', 'Honda', 'Ferrari', 'Toyota', 'BMW', 'Honda', 'Ferrari'] * 100,
    'Type':  ['Sedan', 'SUV', 'Sedan', 'Sport', 'SUV', 'Sedan', 'SUV', 'Sport'] * 100,
    'Origin':['Japan', 'Germany', 'Japan', 'Italy', 'Japan', 'Germany', 'Japan', 'Italy'] * 100,
    'Price': [20000, 60000, 22000, 250000, 25000, 55000, 28000, 260000] * 100
}
df = pd.DataFrame(data)
# 加入雜訊
df['Price'] = df['Price'] + np.random.normal(0, 2000, len(df))

# ==========================================
# 2. 關鍵改變：One-Hot Encoding
# ==========================================
# 使用 pandas 的 get_dummies 自動將類別展開
# columns指定要轉換的欄位
df_processed = pd.get_dummies(df, columns=['Brand', 'Type', 'Origin'])

# 檢查一下現在有多少特徵 (原來的3欄變成了更多欄)
print("One-Hot 特徵欄位:", df_processed.columns.tolist())

X = df_processed.drop('Price', axis=1).values.astype(np.float32)
y = df_processed['Price'].values.astype(np.float32).reshape(-1, 1)

# *技巧*：對目標值(y)進行標準化，這對回歸任務的收斂非常有幫助
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# 轉 Tensor (注意：One-Hot 輸入是 Float，不是 Long)
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

# ==========================================
# 3. 定義 MLP 模型 (標準的全連接層)
# ==========================================
class CarPriceOneHotNet(nn.Module):
    def __init__(self, input_size):
        super(CarPriceOneHotNet, self).__init__()
        
        self.network = nn.Sequential(
            # 第一層輸入就是 One-Hot 展開後的總維度
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), # 標準化層，對於 One-Hot 這種稀疏數據很有幫助
            nn.Dropout(0.2),    # 防止過擬合
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)    # 輸出價格
        )

    def forward(self, x):
        return self.network(x)

# 初始化模型
input_dim = X_train.shape[1] # 自動取得特徵數量
print(f"輸入層維度: {input_dim}") 

model = CarPriceOneHotNet(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 4. 訓練
# ==========================================
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==========================================
# 5. 評估 R-squared
# ==========================================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()
    
    # *關鍵*：要把預測值還原回真實價格範圍，才能算正確的 R2
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_t.numpy())
    
    r2 = r2_score(y_test_original, y_pred_original)

print(f"\nFinal R-squared (One-Hot): {r2:.4f}")


##########
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score
from kan import KAN
import matplotlib.pyplot as plt

# ==========================================
# 1. 生成模擬資料 (類別型輸入 -> 連續型輸出)
# ==========================================
np.random.seed(42)
n_samples = 1000

# 建立兩個類別特徵
data = {
    'Material': np.random.choice(['Wood', 'Metal', 'Plastic'], n_samples),
    'Quality': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Size': np.random.uniform(1, 10, n_samples)  # 混合一個連續變數增加真實感
}
df = pd.DataFrame(data)

# 定義真實的生成函數 (為了測試模型能力)
# 假設：Metal 最貴，High Quality 加成最高，Size 呈非線性關係
def true_price_function(row):
    base_price = 0
    if row['Material'] == 'Wood': base_price += 50
    elif row['Material'] == 'Metal': base_price += 100
    elif row['Material'] == 'Plastic': base_price += 20
    
    multiplier = 1.0
    if row['Quality'] == 'Low': multiplier = 0.8
    elif row['Quality'] == 'Medium': multiplier = 1.0
    elif row['Quality'] == 'High': multiplier = 1.5
    
    # 加上 Size 的非線性關係 (Size^2) 和一些隨機雜訊
    return (base_price * multiplier) + (row['Size'] ** 2) + np.random.normal(0, 5)

df['Price'] = df.apply(true_price_function, axis=1)

print("--- 原始資料預覽 ---")
print(df.head())

# ==========================================
# 2. 特徵工程 (關鍵步驟！)
# ==========================================

# A. 分離特徵與目標
X = df[['Material', 'Quality', 'Size']]
y = df['Price'].values

# B. 類別特徵編碼 (One-Hot Encoding)
# KAN 需要數值輸入，One-Hot 是最保留資訊的方法
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['Material', 'Quality']])
feature_names_encoded = encoder.get_feature_names_out(['Material', 'Quality'])

# 將編碼後的特徵與原始連續特徵合併
X_final = np.hstack([X_encoded, X[['Size']].values])
all_feature_names = list(feature_names_encoded) + ['Size']

# C. 正規化 (Normalization) - KAN 對輸入範圍非常敏感！
# KAN 的 B-Spline 網格通常定義在 [-1, 1] 之間，所以將輸入縮放到此範圍至關重要
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_final)

# D. 轉換為 PyTorch Tensor
# 注意：pykan 需要 dataset 是一個字典格式
train_inputs, test_inputs, train_label, test_label = train_test_split(
    torch.tensor(X_scaled, dtype=torch.float32),
    torch.tensor(y, dtype=torch.float32).unsqueeze(1), # 變成 [N, 1] 形狀
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
# 3. 建立與訓練 KAN 模型
# ==========================================

input_dim = X_final.shape[1]  # 輸入維度 (One-hot 後的特徵數)
output_dim = 1                # 輸出維度 (預測價格)

# 初始化 KAN
# width: [輸入層, 隱藏層, 輸出層]
# grid: 網格大小。數值越大，擬合能力越強 (但也越容易過擬合)。建議從 5 開始。
# k: Spline 的階數，通常設為 3 (Cubic spline)
model = KAN(width=[input_dim, 5, output_dim], grid=5, k=3, seed=42)

print("\n--- 開始訓練 ---")
# 使用 LBFGS 優化器，通常在科學計算/回歸問題上比 Adam 收斂得更好、更精確
# steps: 訓練步數
results = model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.0)

# ==========================================
# 4. 評估 R-squared
# ==========================================

# 預測
pred_train = model(dataset['train_input']).detach().numpy()
pred_test = model(dataset['test_input']).detach().numpy()
y_train = dataset['train_label'].detach().numpy()
y_test = dataset['test_label'].detach().numpy()

# 計算 R2
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)

print("\n" + "="*30)
print(f"訓練集 R-squared: {r2_train:.4f}")
print(f"測試集 R-squared: {r2_test:.4f}")
print("="*30)

# 可視化 KAN (可選)
# model.plot() 
# plt.show()
#######################################
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import r2_score
from kan import KAN
import matplotlib.pyplot as plt

# ==========================================
# 1. 生成資料 (保持與之前相同的邏輯)
# ==========================================
np.random.seed(42)
n_samples = 1000

data = {
    'Material': np.random.choice(['Wood', 'Metal', 'Plastic'], n_samples),
    'Quality': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Size': np.random.uniform(1, 10, n_samples)
}
df = pd.DataFrame(data)

# 真實價格函數
def true_price_function(row):
    base_price = 0
    # 注意這裡的價格是非單調的，考驗模型處理 Label 的能力
    if row['Material'] == 'Wood': base_price += 50
    elif row['Material'] == 'Metal': base_price += 100
    elif row['Material'] == 'Plastic': base_price += 20
    
    multiplier = 1.0
    if row['Quality'] == 'Low': multiplier = 0.8
    elif row['Quality'] == 'Medium': multiplier = 1.0
    elif row['Quality'] == 'High': multiplier = 1.5
    
    return (base_price * multiplier) + (row['Size'] ** 2) + np.random.normal(0, 5)

df['Price'] = df.apply(true_price_function, axis=1)

# ==========================================
# 2. 特徵工程 (改用 Label/Ordinal Encoding)
# ==========================================

X = df[['Material', 'Quality', 'Size']]
y = df['Price'].values

# A. 區分連續與類別特徵
cat_features = ['Material', 'Quality']
cont_features = ['Size']

# B. 使用 OrdinalEncoder (即 Label Encoding)
# 這會將字串轉換為 0, 1, 2...
ordinal_encoder = OrdinalEncoder()
X_cat_encoded = ordinal_encoder.fit_transform(X[cat_features])

# C. 合併特徵
# 現在 Material 是一欄 (0,1,2)，Quality 是一欄 (0,1,2)，Size 是一欄
X_combined = np.hstack([X_cat_encoded, X[cont_features].values])

# D. 正規化 (Normalization) - 絕對關鍵步驟！
# 我們必須將 0,1,2 這樣的整數映射到 KAN 的工作區間 [-1, 1]
# 如果沒有這步，Label Encoding 的效果會極差
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_combined)

# 準備 PyTorch 資料集
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
# 3. 建立 KAN 模型
# ==========================================

# 輸入維度現在變小了：
# 1 (Material) + 1 (Quality) + 1 (Size) = 3
input_dim = X_scaled.shape[1] 
output_dim = 1

# 設定 KAN
# 這裡 grid 可以稍微設大一點 (例如 8 或 10)
# 因為模型需要用單一函數去擬合離散的跳躍點 (例如從 input=-1 到 input=0 價格劇烈變化)
model = KAN(width=[input_dim, 5, output_dim], grid=10, k=3, seed=42)

print(f"輸入特徵維度: {input_dim}") 
# 這裡會印出 3，若是 One-Hot 則會是 ~7

print("\n--- 開始訓練 (Label Encoding 版本) ---")
# 訓練
results = model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01)

# ==========================================
# 4. 評估與觀察
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

# 可選：如果您想看模型如何處理 Label Encode 的非線性關係
# model.plot()
# plt.show()


#########################
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score
from kan import KAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
# ==========================================
# 1. 生成模擬資料 (類別型輸入 -> 連續型輸出)
# ==========================================
np.random.seed(42)
n_samples = 1000

# 建立兩個類別特徵
data = {
    'Material': np.random.choice(['Wood', 'Metal', 'Plastic'], n_samples),
    'Quality': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Size': np.random.uniform(1, 10, n_samples)  # 混合一個連續變數增加真實感
}
df = pd.DataFrame(data)

# 定義真實的生成函數 (為了測試模型能力)
# 假設：Metal 最貴，High Quality 加成最高，Size 呈非線性關係
def true_price_function(row):
    base_price = 0
    if row['Material'] == 'Wood': base_price += 50
    elif row['Material'] == 'Metal': base_price += 100
    elif row['Material'] == 'Plastic': base_price += 20
    
    multiplier = 1.0
    if row['Quality'] == 'Low': multiplier = 0.8
    elif row['Quality'] == 'Medium': multiplier = 1.0
    elif row['Quality'] == 'High': multiplier = 1.5
    
    # 加上 Size 的非線性關係 (Size^2) 和一些隨機雜訊
    return (base_price * multiplier) + (row['Size'] ** 2) + np.random.normal(0, 5)

df['Price'] = df.apply(true_price_function, axis=1)

print("--- 原始資料預覽 ---")
print(df.head())

# ==========================================
# 2. 特徵工程 (關鍵步驟！)
# ==========================================

# A. 分離特徵與目標
X = df[['Material', 'Quality', 'Size']]
y = df['Price'].values

# # B. 類別特徵編碼 (One-Hot Encoding)
# # KAN 需要數值輸入，One-Hot 是最保留資訊的方法
# encoder = OneHotEncoder(sparse_output=False)
# X_encoded = encoder.fit_transform(X[['Material', 'Quality']])
# feature_names_encoded = encoder.get_feature_names_out(['Material', 'Quality'])

# # 將編碼後的特徵與原始連續特徵合併
# X_final = np.hstack([X_encoded, X[['Size']].values])
# all_feature_names = list(feature_names_encoded) + ['Size']



# B. 使用 OrdinalEncoder (即 Label Encoding)
# 這會將字串轉換為 0, 1, 2...
cat_features = ['Material', 'Quality']
cont_features = ['Size']

ordinal_encoder = OrdinalEncoder()
X_cat_encoded = ordinal_encoder.fit_transform(X[cat_features])

# C. 合併特徵
# 現在 Material 是一欄 (0,1,2)，Quality 是一欄 (0,1,2)，Size 是一欄
X_combined = np.hstack([X_cat_encoded, X[cont_features].values])

# D. 正規化 (Normalization) - 絕對關鍵步驟！
# 我們必須將 0,1,2 這樣的整數映射到 KAN 的工作區間 [-1, 1]
# 如果沒有這步，Label Encoding 的效果會極差
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_combined)



# C. 正規化 (Normalization) - KAN 對輸入範圍非常敏感！
# KAN 的 B-Spline 網格通常定義在 [-1, 1] 之間，所以將輸入縮放到此範圍至關重要
# scaler = MinMaxScaler(feature_range=(-1, 1))
# X_scaled = scaler.fit_transform(X_final)

# D. 轉換為 PyTorch Tensor
# 注意：pykan 需要 dataset 是一個字典格式
train_inputs, test_inputs, train_label, test_label = train_test_split(
    torch.tensor(X_scaled, dtype=torch.float32),
    torch.tensor(y, dtype=torch.float32).unsqueeze(1), # 變成 [N, 1] 形狀
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
# 3. 建立與訓練 KAN 模型
# ==========================================

input_dim = X_combined.shape[1]  # 輸入維度 (One-hot 後的特徵數)
output_dim = 1                # 輸出維度 (預測價格)

# 初始化 KAN
# width: [輸入層, 隱藏層, 輸出層]
# grid: 網格大小。數值越大，擬合能力越強 (但也越容易過擬合)。建議從 5 開始。
# k: Spline 的階數，通常設為 3 (Cubic spline)
model = KAN(width=[input_dim, 5, output_dim], grid=5, k=3, seed=42)

print("\n--- 開始訓練 ---")
# 使用 LBFGS 優化器，通常在科學計算/回歸問題上比 Adam 收斂得更好、更精確
# steps: 訓練步數
results = model.fit(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.0)

# ==========================================
# 4. 評估 R-squared
# ==========================================

# 預測
pred_train = model(dataset['train_input']).detach().numpy()
pred_test = model(dataset['test_input']).detach().numpy()
y_train = dataset['train_label'].detach().numpy()
y_test = dataset['test_label'].detach().numpy()

# 計算 R2
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)

print("\n" + "="*30)
print(f"訓練集 R-squared: {r2_train:.4f}")
print(f"測試集 R-squared: {r2_test:.4f}")
print("="*30)

# 可視化 KAN (可選)
# model.plot() 
# plt.show()
