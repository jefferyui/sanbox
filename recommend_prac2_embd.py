# python tf 使用sns dataset 做一個深度學習NCF 推薦系統
# dataset KeyError: 'user_id'
# 將所有欄位除了rating以外，做完資料前處理後全部丟入model
# Error: model.fit(train_inputs, y_train, epochs=10, batch_size=32, validation_split=0.2) ValueError: Layer "functional_4" expects 4 input(s), but it received 5 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(None, 1) dtype=int32>, <tf.Tensor 'data_1:0' shape=(None, 1) dtype=float32>, <tf.Tensor 'data_2:0' shape=(None, 1) dtype=int32>, <tf.Tensor 'data_3:0' shape=(None, 1) dtype=int32>, <tf.Tensor 'data_4:0' shape=(None, 1) dtype=float32>]
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# 對類別型特徵進行標籤編碼 (Label Encoding)
label_encoders = {}
for column in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le  # 保存編碼器以供後續使用

# 對數值型特徵進行標準化
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 構建模型
input_layers = []
embedding_layers = []

# 處理類別型特徵
for column in X.select_dtypes(include=['int32', 'int64']).columns:
    n_unique_values = X[column].nunique()
    input_layer = Input(shape=(1,), name=column)
    embedding_layer = Embedding(input_dim=n_unique_values, output_dim=8, input_length=1)(input_layer)
    embedding_layer = Flatten()(embedding_layer)
    input_layers.append(input_layer)
    embedding_layers.append(embedding_layer)

# 處理數值型特徵
numeric_input = Input(shape=(len(numeric_features),), name='numeric_input')
input_layers.append(numeric_input)
embedding_layers.append(numeric_input)

# 結合所有嵌入層與數值型特徵
combined = Concatenate()(embedding_layers)

# 增加深度神經網絡層
dense = Dense(128, activation='relu')(combined)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='linear')(dense)  # 使用線性激活函數以適應回歸問題

# 定義和編譯模型
model = Model(inputs=input_layers, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # 使用均方誤差作為損失函數

# 準備輸入數據
train_inputs = [X_train[column] for column in X.select_dtypes(include=['int32', 'int64']).columns]
train_inputs.append(X_train[numeric_features])

test_inputs = [X_test[column] for column in X.select_dtypes(include=['int32', 'int64']).columns]
test_inputs.append(X_test[numeric_features])

# 訓練模型
model.fit(train_inputs, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 預測
y_pred = model.predict(test_inputs)

# 計算 R²
r_squared = r2_score(y_test, y_pred)
print(f"R² Score: {r_squared}")

