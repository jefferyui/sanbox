# ==========================================================
# Deep Learning → AND Rule Extraction
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ==========================================================
# 1. 建立資料
# ==========================================================
def create_dataset():
    data = pd.DataFrame({
        "Age": [45, 50, 62, 30, 25, 55, 40, 65, 48, 52, 38, 60],
        "BMI": [28, 31, 35, 22, 20, 33, 27, 36, 29, 34, 26, 37],
        "BP": [140, 150, 160, 120, 115, 155, 130, 170, 145, 158, 128, 165],
        "Smoker": [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        "Disease": [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    return data


# ==========================================================
# 2. 建立神經網路
# ==========================================================
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# ==========================================================
# 3. 訓練模型
# ==========================================================
def train_model(model, X_train, y_train):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(300):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    print("✅ Neural Network Training Complete")


# ==========================================================
# 4. AND Rule Extraction
# ==========================================================
def extract_and_rule(data, model, scaler):

    X = data.drop("Disease", axis=1)
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32))
        preds = (preds.numpy() > 0.5).astype(int).flatten()

    positive_samples = data[preds == 1]

    print("\n✅ Samples Predicted as Disease=1:")
    print(positive_samples)

    # 取所有正樣本的條件交集（min threshold approach）
    rule_conditions = {}

    for col in ["Age", "BMI", "BP"]:
        min_value = positive_samples[col].min()
        rule_conditions[col] = f"{col} >= {min_value}"

    # 處理二元變數
    if positive_samples["Smoker"].all() == 1:
        rule_conditions["Smoker"] = "Smoker = 1"

    print("\n✅ Extracted AND Rule:\n")
    print("IF ", " AND ".join(rule_conditions.values()))
    print("THEN Disease = 1")


# ==========================================================
# 5. 主程式
# ==========================================================
def main():

    data = create_dataset()

    X = data.drop("Disease", axis=1)
    y = data["Disease"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

    model = SimpleNN(input_dim=X.shape[1])

    train_model(model, X_train_tensor, y_train_tensor)

    extract_and_rule(data, model, scaler)


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    main()
