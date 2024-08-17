# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:01:17 2024

"""
# python tf 使用sns dataset 做一個深度學習NCF 推薦系統
# dataset KeyError: 'user_id'
# 將所有欄位除了rating以外，做完資料前處理後全部丟入model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd

# 加載資料集
data = sns.load_dataset('fmri')

# 模擬生成 rating 欄位
data['rating'] = data['signal'] / data['signal'].max()  # 正規化 rating 介於 [0, 1] 之間

# 分割 features 和 target (rating)
X = data.drop(columns=['rating'])
y = data['rating']

# 定義資料前處理管道
# 對類別型欄位進行 One-Hot 編碼，對數值型欄位進行標準化
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# 將前處理管道應用到資料集
X_processed = preprocessor.fit_transform(X)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 構建模型
input_shape = X_train.shape[1]
inputs = Input(shape=(input_shape,))
dense = Dense(128, activation='relu')(inputs)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='linear')(dense)  # 使用線性激活函數以適應回歸問題

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # 使用均方誤差作為損失函數

# 訓練模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 預測
y_pred = model.predict(X_test)

# 計算 R²
r_squared = r2_score(y_test, y_pred)
print(f"R² Score: {r_squared}")
