# -*- coding: utf-8 -*-

# python tf 使用sns dataset 做一個深度學習NCF 推薦系統
# dataset KeyError: 'user_id'
# 將所有欄位除了rating以外，做完資料前處理後全部丟入model
# 使用推薦系統雙塔model
# 將所有欄位除了rating以外，做完資料前處理後全部丟入model
# ValueError: Layer "functional_8" expects 1 input(s), but it received 5 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(None,) dtype=float32>, <tf.Tensor 'data_1:0' shape=(None,) dtype=float32>, <tf.Tensor 'data_2:0' shape=(None,) dtype=float32>, <tf.Tensor 'data_3:0' shape=(None,) dtype=float32>, <tf.Tensor 'data_4:0' shape=(None,) dtype=float32>]
# help dedug for 
# 你所遇到的問題可能是由於模型的輸入層定義和實際提供的數據不匹配。從你提供的程式碼來看，你有四個輸入層（user_input、item_input、numeric_input、categorical_input），但在模型構建中，你只定義了三個輸入層，這會導致錯誤。
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate, Dot
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd

# 加載資料集
data = sns.load_dataset('fmri')

# 模擬生成 rating 欄位和 user_id, item_id 欄位
data['rating'] = data['signal'] / data['signal'].max()  # 正規化 rating 介於 [0, 1] 之間
data['user_id'] = data['subject']
data['item_id'] = data['region']

# 分割 features 和 target (rating)
X = data.drop(columns=['rating'])
y = data['rating']

# 對類別型特徵進行有序編碼 (Ordinal Encoding)
ordinal_encoder = OrdinalEncoder()
X[['user_id', 'item_id']] = ordinal_encoder.fit_transform(X[['user_id', 'item_id']])

# 對數值型特徵進行標準化
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

categorical_features = X.select_dtypes(include=['object', 'category']).columns
# 對類別型特徵進行標籤編碼 (Label Encoding)
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le  # 保存編碼器以供後續使用
# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定義用戶塔
user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(input_dim=X['user_id'].nunique() + 1, output_dim=8)(user_input)
user_vector = Flatten()(user_embedding)
user_dense = Dense(64, activation='relu')(user_vector)

# 定義物品塔
item_input = Input(shape=(1,), name='item_input')
item_embedding = Embedding(input_dim=X['item_id'].nunique() + 1, output_dim=8)(item_input)
item_vector = Flatten()(item_embedding)
item_dense = Dense(64, activation='relu')(item_vector)

# 處理數值型特徵
numeric_input = Input(shape=(len(numeric_features),), name='numeric_input')
numeric_dense = Dense(64, activation='relu')(numeric_input)


# 處理類別型特徵
categorical_input = Input(shape=(len(categorical_features),), name='categorical_input')
categorical_dense = Dense(64, activation='relu')(categorical_input)

# 結合用戶塔和物品塔的輸出
concat = Concatenate()([user_dense, item_dense, numeric_dense, categorical_dense])
output = Dense(1, activation='linear')(concat)  # 使用線性激活函數以適應回歸問題

# 構建模型
model = Model(inputs=[user_input, item_input, numeric_input, categorical_input ], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 準備輸入數據
train_user_input = X_train[['user_id']].values
train_item_input = X_train[['item_id']].values
train_numeric_input = X_train[numeric_features].values
train_categorical_input = X_train[categorical_features].values

test_user_input = X_test[['user_id']].values
test_item_input = X_test[['item_id']].values
test_numeric_input = X_test[numeric_features].values
test_categorical_input = X_test[categorical_features].values
# 訓練模型
model.fit([train_user_input, train_item_input, train_numeric_input, train_categorical_input], y_train, epochs=10, batch_size=32, validation_split=0.2)

# 預測
y_pred = model.predict([test_user_input, test_item_input, test_numeric_input, test_categorical_input])

# 計算 R²
r_squared = r2_score(y_test, y_pred)
print(f"R² Score: {r_squared}")
