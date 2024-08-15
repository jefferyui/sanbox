# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:58:51 2024

@author: jeffery
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 21:04:39 2024

@author: jeffery
"""
# 使用tf.feature_column 處理data，並重作一次
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import LabelEncoder
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


# 编码类别特征
label_encoders = {}
for col in ['sex', 'embarked', 'pclass']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# 构建输入字典
X_train_dict = {name: X_train[name].values for name in X_train.columns}
X_test_dict = {name: X_test[name].values for name in X_test.columns}


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

class AgentAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(AgentAttention, self).__init__(**kwargs)
        self.units = units
        self.query_dense = layers.Dense(units)
        self.key_dense = layers.Dense(units)
        self.value_dense = layers.Dense(units)
        self.softmax = layers.Softmax(axis=-1)

    def call(self, inputs):
        # 输入分为 Query, Key, Value
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # 计算注意力权重
        scores = tf.matmul(query, key, transpose_b=True)
        attention_weights = self.softmax(scores)
        
        # 对值进行加权求和
        context_vector = tf.matmul(attention_weights, value)
        return context_vector

    def get_config(self):
        config = super(AgentAttention, self).get_config()
        config.update({"units": self.units})
        return config
    
class AgentAttentionBlock(layers.Layer):
    def __init__(self, units, num_heads, ff_dim, **kwargs):
        super(AgentAttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # 多头自注意力机制
        self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)
        self.attention_dropout = layers.Dropout(0.3)

        # 前馈网络
        self.dense1 = layers.Dense(ff_dim, activation='relu')
        self.dense2 = layers.Dense(units)
        self.ffn_dropout = layers.Dropout(0.3)

        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        # Self-Attention with Multi-Head Attention
        attn_output = self.multi_head_attention(inputs, inputs)
        attn_output = self.attention_dropout(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed Forward Network
        ffn_output = self.dense1(out1)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(AgentAttentionBlock, self).get_config()
        config.update({
            "units": self.units,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config
def build_model_with_embeddings_and_agent_attention(num_heads=2, ff_dim=32):
    # 输入层
    inputs = {}
    embedding_layers = []
    
    # 数值特征的输入层
    for name in ['age', 'fare']:
        inputs[name] = tf.keras.Input(name=name, shape=(1,), dtype=tf.float32)
    
    # 类别特征的嵌入层
    for name in ['sex', 'embarked', 'pclass']:
        vocab_size = len(X_train[name].unique()) + 1  # 加1是为了包含未知类别
        embedding_dim = min(50, vocab_size // 2)  # 设置嵌入维度
        inputs[name] = tf.keras.Input(name=name, shape=(1,), dtype=tf.float32)
        embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1)(inputs[name])
        embedding = layers.Reshape(target_shape=(embedding_dim,))(embedding)
        embedding_layers.append(embedding)
    
    # 将数值特征和嵌入特征连接
    all_features = layers.Concatenate()(list(inputs.values()) + embedding_layers)
    
    # 重塑输入数据以适应1D卷积
    x = layers.Reshape((-1, 1))(all_features)

    # 添加小波卷积层
    # x = WaveletConv(filters=32, kernel_size=3)(x)

    # 全连接层，加入Dropout和Batch Normalization
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Reshape for Attention Block
    x = layers.Reshape((1, -1))(x)

    # 添加 Agent Attention Block
    # x = AgentAttentionBlock(units=32, num_heads=num_heads, ff_dim=ff_dim)(x)

    # Transformer Blocks
    for _ in range(1):  # 只添加一个 Transformer Block
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)
    
    x = layers.Flatten()(x)

    # 输出层
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # 定义模型
    model = models.Model(inputs=inputs, outputs=output)
    
    return model

# 构建并编译模型
# 构建并编译模型
model = build_model_with_embeddings_and_agent_attention()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])

# 显示模型架构
model.summary()

# 训练模型，使用早停策略
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train_dict, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

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


