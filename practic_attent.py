# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 22:00:05 2024

@author: jeffery
"""

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Attention, Concatenate, Input

# 載入 Titanic 資料集
titanic = sns.load_dataset('titanic')

# 預處理資料
titanic = titanic.dropna(subset=['age', 'embarked', 'sex', 'class'])
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
titanic['class'] = titanic['class'].map({'First': 0, 'Second': 1, 'Third': 2})

X = titanic[['age', 'sex', 'embarked', 'class']]
y = titanic['survived']

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
def create_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = tf.keras.layers.Reshape((1, 32))(x)
    
    # 注意力機制
    attention = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(attention)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# 建立模型
model = create_attention_model((X_train.shape[1],))
model.summary()

# 訓練模型
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
