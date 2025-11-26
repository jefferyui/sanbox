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
