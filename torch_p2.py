# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 22:22:56 2024

@author: jeffery
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# 1. 加载并预处理数据
data = sns.load_dataset('titanic')

# 处理缺失值
data = data.drop(['deck', 'embark_town', 'alive'], axis=1)  # 删除缺失值较多的列
data = data.dropna()  # 删除包含 NaN 的行

# 编码分类特征
label_encoders = {}
for column in ['sex', 'embarked', 'class', 'who', 'adult_male', 'alone']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 特征和标签
X = data.drop(['survived'], axis=1)
y = data['survived']

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. 构建模型
class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        
        # Transformer block
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))  # 先通过线性层将输入维度调整为 64
        
        # Transformer expects input of shape (seq_len, batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, seq_len, d_model) to (batch_size, 1, d_model)
        x = x.permute(1, 0, 2)  # (1, batch_size, d_model)
        
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).squeeze(1)  # Convert back to (batch_size, d_model)
        
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x

model = TitanicModel()

# 3. 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round()
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f'Accuracy: {accuracy:.4f}')
