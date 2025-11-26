import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class BostonMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
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
