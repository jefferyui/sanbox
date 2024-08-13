# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:05:23 2024

@author: jeffery
"""
# 使用tf.feature_column 處理data，並重作一次
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# 加载 Titanic 数据集
df = sns.load_dataset('titanic')

# 处理缺失值
df = df.dropna(subset=['age', 'fare', 'embarked', 'sex', 'pclass', 'survived'])

df['sex'] = df['sex'].astype(str)
df['embarked'] = df['embarked'].astype(str)
df['pclass'] = df['pclass'].astype(str)  

# 选择特征和目标
features = ['pclass', 'sex', 'age', 'fare', 'embarked']
target = 'survived'

X = df[features]
y = df[target]

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 定义数值特征列
age = tf.feature_column.numeric_column('age')
fare = tf.feature_column.numeric_column('fare')

# 定义类别特征列，并确保类型是字符串
pclass = tf.feature_column.categorical_column_with_vocabulary_list('pclass', df['pclass'].unique())
sex = tf.feature_column.categorical_column_with_vocabulary_list('sex', df['sex'].unique())
embarked = tf.feature_column.categorical_column_with_vocabulary_list('embarked', df['embarked'].unique())

# 将类别特征转换为指示符列
pclass_one_hot = tf.feature_column.indicator_column(pclass)
sex_one_hot = tf.feature_column.indicator_column(sex)
embarked_one_hot = tf.feature_column.indicator_column(embarked)

# 创建特征列列表
feature_columns = [age, fare, pclass_one_hot, sex_one_hot, embarked_one_hot]

# 创建 DenseFeatures 输入层
feature_layer = layers.DenseFeatures(feature_columns)


# 将训练和测试数据转换为字典格式
X_train_dict = {name: np.array(value) for name, value in X_train.items()}
X_test_dict = {name: np.array(value) for name, value in X_test.items()}


# 定义 Transformer Block
def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    ff_output = layers.Dense(ff_dim, activation='relu')(attention_output)
    ff_output = layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    transformer_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
    return transformer_output


# model 準確率反而降低，要怎麼改model
# model 加入小波卷积 重做一次
import tensorflow as tf
from tensorflow.keras import layers

# class WaveletConv(layers.Layer):
#     def __init__(self, filters, kernel_size, **kwargs):
#         super(WaveletConv, self).__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.conv_low = layers.Conv1D(filters, kernel_size, padding='same')
#         self.conv_high = layers.Conv1D(filters, kernel_size, padding='same')

#     def call(self, inputs):
#         # 小波变换的低频部分
#         low_freq = self.conv_low(inputs)
#         # 小波变换的高频部分
#         high_freq = inputs - low_freq
#         return tf.concat([low_freq, high_freq], axis=-1)

#     def get_config(self):
#         config = super(WaveletConv, self).get_config()
#         config.update({"filters": self.filters, "kernel_size": self.kernel_size})
#         return config

# # 定义 Transformer Block
# def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
#     attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
#     attention_output = layers.Dropout(dropout_rate)(attention_output)
#     attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

#     ff_output = layers.Dense(ff_dim, activation='relu')(attention_output)
#     ff_output = layers.Dense(inputs.shape[-1])(ff_output)
#     ff_output = layers.Dropout(dropout_rate)(ff_output)
#     transformer_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
#     return transformer_output


# # 定义模型，包括小波卷积层
# def build_model(feature_layer, num_heads=8, ff_dim=32):
#     # 输入层
#     inputs = {}
#     for name in X_train.columns:
#         if name in ['age', 'fare']:  # 数值特征
#             inputs[name] = tf.keras.Input(name=name, shape=(), dtype=tf.float32)
#         else:  # 类别特征
#             inputs[name] = tf.keras.Input(name=name, shape=(), dtype=tf.string)
    
#     # 特征层
#     x = feature_layer(inputs)
#     # 添加一些全连接层
    
#     x = layers.Dense(32, activation='relu')(x)
#     x = layers.Dense(64, activation='relu')(x)
#     # 重塑输入数据以适应1D卷积
#     x = layers.Reshape((-1, 1))(x)

#     # 添加小波卷积层
#     x = WaveletConv(filters=64, kernel_size=3)(x)

#     # 全连接层
#     x = layers.Flatten()(x)
#     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dense(32, activation='relu')(x)

#     # Reshape for Transformer
#     x = layers.Reshape((1, -1))(x)

#     # Transformer Blocks
#     for _ in range(2):
#         x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)
    
#     x = layers.Flatten()(x)
#     # Add final hidden layers
#     x = layers.Dense(64, activation='relu')(x)
#     x = layers.Dense(32, activation='relu')(x)
#     # 输出层
#     output = layers.Dense(1, activation='sigmoid')(x)
    
#     # 定义模型
#     model = models.Model(inputs=inputs, outputs=output)
    
#     return model

# # 创建 DenseFeatures 输入层
# feature_layer = layers.DenseFeatures(feature_columns)

# # 构建并编译模型
# model = build_model(feature_layer)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # 显示模型架构
# model.summary()

# # 训练模型
# model.fit(X_train_dict, y_train, epochs=50, batch_size=32, validation_split=0.2)

# # 在测试集上评估模型
# loss, accuracy = model.evaluate(X_test_dict, y_test)
# print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


# 调整小波卷积层的参数，并加入Batch Normalization和Dropout
class WaveletConv(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(WaveletConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_low = layers.Conv1D(filters, kernel_size, padding='same')
        self.conv_high = layers.Conv1D(filters, kernel_size, padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs):
        # 小波变换的低频部分
        low_freq = self.conv_low(inputs)
        # 小波变换的高频部分
        high_freq = inputs - low_freq
        x = tf.concat([low_freq, high_freq], axis=-1)
        x = self.batch_norm(x)
        return self.dropout(x)

    def get_config(self):
        config = super(WaveletConv, self).get_config()
        config.update({"filters": self.filters, "kernel_size": self.kernel_size})
        return config

# 构建简化的模型，调整学习率，加入Dropout和Batch Normalization
def build_model(feature_layer, num_heads=8, ff_dim=32):
    # 输入层
    inputs = {}
    for name in X_train.columns:
        if name in ['age', 'fare']:  # 数值特征
            inputs[name] = tf.keras.Input(name=name, shape=(), dtype=tf.float32)
        else:  # 类别特征
            inputs[name] = tf.keras.Input(name=name, shape=(), dtype=tf.string)
    
    # 特征层
    x = feature_layer(inputs)
#     # 添加一些全连接层
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)    
    # 重塑输入数据以适应1D卷积
    x = layers.Reshape((-1, 1))(x)

    # 添加小波卷积层
    x = WaveletConv(filters=64, kernel_size=3)(x)

    # 全连接层，加入Dropout和Batch Normalization
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Reshape for Transformer
    x = layers.Reshape((1, -1))(x)

    # Transformer Blocks
    for _ in range(2):  # 减少Transformer Block的数量
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)
    
    x = layers.Flatten()(x)

    # 输出层
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # 定义模型
    model = models.Model(inputs=inputs, outputs=output)
    
    return model

# 调整优化器和学习率
model = build_model(feature_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型，使用早停策略
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train_dict, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_test_dict, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


# 进行预测
predictions = model.predict(X_test_dict)

# 将概率值转换为二进制结果
predictions_binary = (predictions > 0.5).astype(int)

# 显示前5个预测结果
for i in range(5):
    print(f"Passenger {i+1}: Predicted Survival = {predictions_binary[i][0]}, Actual Survival = {y_test.iloc[i]}")
