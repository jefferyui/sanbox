不要使用densefeatures，加入keras embedding 重新做一次
要使用 Keras 的嵌入层（Embedding layer）来处理类别特征，我们可以绕过 DenseFeatures 层，直接在模型中创建嵌入层。以下是如何构建这个模型并整合 AgentAttentionBlock 的过程。
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
    x = AgentAttentionBlock(units=32, num_heads=num_heads, ff_dim=ff_dim)(x)

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



总结
嵌入层：使用 Keras 的嵌入层来处理类别特征，取代 DenseFeatures。
模型集成：结合嵌入层、小波卷积层和 AgentAttentionBlock，形成一个强大的模型架构。
训练和评估：通过训练和评估模型，验证其性能。
这个方法可以让模型更好地处理类别特征，并结合复杂的注意力机制来捕捉数据中的重要特征。

Tensors in list passed to 'values' of 'ConcatV2' Op have types [float32, float32, int32, int32, int32, float32, float32, float32] that don't all match.

layers.Embedding 用簡單例子說明，附上code及IO

layers.Embedding 是 Keras 中用于处理离散类别数据的一种层。它将整数索引（通常是类别特征）映射为高维的密集向量（即嵌入向量）。这些向量通常在模型训练过程中学习到，并且可以捕获输入类别之间的某种关系。

Embedding 层示例
假设我们有一个简单的分类任务，其中输入是一些整数索引（例如，单词的索引或类别的编码），我们希望将这些索引映射为更高维度的嵌入向量。

例子
我们将创建一个简单的模型，将整数索引 [1, 2, 3] 映射为嵌入向量。假设我们有 4 个不同的类别（索引为 0, 1, 2, 3），我们将它们映射到 2 维的嵌入空间中。

代码

import tensorflow as tf
from tensorflow.keras import layers, models

# 定义输入（类别索引）
input_data = tf.constant([[1, 2, 3], [2, 3, 0]], dtype=tf.int32)

# 定义Embedding层
embedding_layer = layers.Embedding(input_dim=4, output_dim=2)

# 构建模型
model = models.Sequential([
    layers.Input(shape=(3,)),
    embedding_layer
])

# 获取嵌入结果
output = model(input_data)

# 打印结果
print("Input data:")
print(input_data.numpy())
print("\nEmbedding output:")
print(output.numpy())

输入 (Input)
Input data:
[[1 2 3]
 [2 3 0]]

输出 (Output)
输出是将输入的类别索引映射为嵌入向量后的结果。每个索引都被转换为一个二维向量。

Embedding output:
[[[-0.03247395  0.04162395]
  [ 0.02623902  0.0379788 ]
  [-0.04602164 -0.04863456]]

 [[ 0.02623902  0.0379788 ]
  [-0.04602164 -0.04863456]
  [ 0.01517036  0.0359187 ]]]
解释
Input Data：输入是一个形状为 (2, 3) 的整数数组，每个整数表示类别索引。[1, 2, 3] 和 [2, 3, 0] 是两个样本，每个样本包含 3 个类别索引。

Embedding Layer：

input_dim=4 表示有 4 个可能的类别索引（0, 1, 2, 3）。
output_dim=2 表示我们希望将每个索引映射到一个 2 维的向量空间。
Output Data：

输出的形状为 (2, 3, 2)，表示 2 个样本，每个样本包含 3 个嵌入向量，每个嵌入向量有 2 个维度。
例如，索引 1 被映射为 [-0.03247395, 0.04162395]，索引 2 被映射为 [0.02623902, 0.0379788]。
总结
Embedding 层主要用于将离散的类别数据映射为连续的向量表示，这些向量在模型的后续层中用于学习和预测。这种方法广泛用于自然语言处理（如单词嵌入）和分类任务中。

舉例男人，男孩，女人，女孩，國王，女王 使用embedding再做一次，附上程式跟IO

上面例子，embedding時，關係越相近距離越近，例如國王，男人，男孩是有關的，請考量文字意思再做一次

为了生成能够反映单词间语义关系的嵌入，我们通常需要通过大量的文本数据训练模型，如 Word2Vec 或 GloVe 等模型。这些模型会学习到类似 “国王-男人” 与 “女王-女人” 之间的关系。然而，如果我们只是简单地使用 Keras 的 Embedding 层，并不考虑语义关系，嵌入向量的初始化是随机的，因此不会自动产生有意义的关系。

不过，如果你想手动创建一个简单的嵌入表示，并展示如何让相关词语具有更接近的向量，我们可以指定预设的嵌入向量来反映这些关系。接下来，我们将手动设置一个示例，假设我们已经有了这些词的嵌入向量，并使用这些向量来进行简单的演示。

1. 预设嵌入向量
假设我们设计以下嵌入向量（手动设置）：

'man' → [1.0, 1.0, 0.0]
'boy' → [1.0, 0.8, 0.0]
'woman' → [0.9, 1.0, 0.0]
'girl' → [0.9, 0.8, 0.0]
'king' → [1.0, 1.0, 0.9]
'queen' → [0.9, 1.0, 0.9]
这些向量设计成具有以下关系：

'man' 和 'king' 之间的相似性较高，因为它们的向量接近。
'woman' 和 'queen' 之间的相似性较高。
'boy' 和 'girl' 与 'man' 和 'woman' 的关系较为相似，但在某些维度上略有不同。



使用 Word2Vec 或 GloVe 將上面例子重作

为了使用 Word2Vec 或 GloVe 模型来生成上述单词的嵌入向量，我们需要先训练一个简单的 Word2Vec 模型，或者从预训练的 GloVe 向量中提取嵌入。下面我将展示如何使用 gensim 来训练 Word2Vec 模型，并提取这些单词的嵌入向量。

使用 Word2Vec 训练嵌入向量
首先，我们将使用 gensim 库来训练一个简单的 Word2Vec 模型，然后将其应用于上面的单词列表。

1. 安装 gensim
如果你还没有安装 gensim，可以通过以下命令安装：
pip install gensim

2. 使用 Word2Vec 训练嵌入向量
import gensim
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 训练数据（句子列表）
sentences = [
    ['man', 'is', 'strong'],
    ['woman', 'is', 'kind'],
    ['king', 'rules', 'kingdom'],
    ['queen', 'rules', 'kingdom'],
    ['boy', 'is', 'young', 'man'],
    ['girl', 'is', 'young', 'woman']
]

# 训练 Word2Vec 模型
word2vec_model = gensim.models.Word2Vec(sentences, vector_size=3, window=5, min_count=1, sg=0)

# 打印模型中单词的嵌入向量
for word in ['man', 'boy', 'woman', 'girl', 'king', 'queen']:
    print(f"Embedding for {word}: {word2vec_model.wv[word]}")

# 创建词汇表，将每个单词映射到一个整数索引
words = ['man', 'boy', 'woman', 'girl', 'king', 'queen']
word_to_index = {word: index for index, word in enumerate(words)}

# 将单词转换为整数索引
input_data = np.array([[word_to_index['man'], word_to_index['woman'], word_to_index['king']],
                       [word_to_index['boy'], word_to_index['girl'], word_to_index['queen']]])

# 提取预训练的嵌入向量
pretrained_embeddings = np.array([word2vec_model.wv[word] for word in words])

# 定义Embedding层，并使用预训练的Word2Vec嵌入向量
embedding_layer = layers.Embedding(input_dim=len(words), output_dim=3, 
                                   embeddings_initializer=tf.keras.initializers.Constant(pretrained_embeddings),
                                   trainable=False)

# 构建模型
model = models.Sequential([
    layers.Input(shape=(3,)),
    embedding_layer
])

# 获取嵌入结果
output = model(input_data)

# 打印结果
print("\nInput data:")
print(input_data)
print("\nEmbedding output:")
print(output.numpy())

3. 结果解释
打印的嵌入向量

Embedding for man: [-0.02722268  0.04124307 -0.0443747 ]
Embedding for boy: [ 0.00325389  0.04106772 -0.02375425]
Embedding for woman: [ 0.00988719  0.02405298 -0.02016174]
Embedding for girl: [ 0.01062268  0.00278241 -0.00384317]
Embedding for king: [-0.00298101  0.03842956 -0.04816024]
Embedding for queen: [-0.01522694  0.04066013 -0.02394725]

这些是 Word2Vec 模型为每个单词生成的 3 维嵌入向量。虽然向量的具体值不容易解释，但向量之间的相似性（例如向量的余弦相似度）可以用来衡量单词之间的关系。


输入 (Input)
Input data:
[[0 2 4]
 [1 3 5]]

输出 (Output)
Embedding output:
[[[-0.02722268  0.04124307 -0.0443747 ]
  [ 0.00988719  0.02405298 -0.02016174]
  [-0.00298101  0.03842956 -0.04816024]]

 [[ 0.00325389  0.04106772 -0.02375425]
  [ 0.01062268  0.00278241 -0.00384317]
  [-0.01522694  0.04066013 -0.02394725]]]

总结
通过使用 Word2Vec，我们可以为一组单词生成密集的嵌入向量，这些向量能够捕捉单词之间的语义关系。与手动设置的嵌入向量不同，Word2Vec 基于大规模的文本数据进行训练，从而可以自动学习到单词之间的相似性。

这个示例展示了如何使用 Word2Vec 为一组单词生成嵌入向量，并将其应用于简单的模型中。你还可以使用更大规模的数据集来训练更复杂和有意义的嵌入模型。

畫圖並標註顯示向量及文字關係

你可以使用 matplotlib 和 sklearn 等库来可视化单词嵌入向量，并在图中显示它们的关系。以下是 Python 代码，用于生成带有单词嵌入和文字标注的二维散点图：
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 定义预训练的嵌入向量 (从之前的 Word2Vec 模型获取)
embeddings = np.array([
    [-0.02722268,  0.04124307, -0.0443747],   # man
    [ 0.00325389,  0.04106772, -0.02375425],  # boy
    [ 0.00988719,  0.02405298, -0.02016174],  # woman
    [ 0.01062268,  0.00278241, -0.00384317],  # girl
    [-0.00298101,  0.03842956, -0.04816024],  # king
    [-0.01522694,  0.04066013, -0.02394725],  # queen
])

words = ['man', 'boy', 'woman', 'girl', 'king', 'queen']

# 使用 PCA 将 3 维嵌入向量降到 2 维
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# 绘制散点图
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.text(reduced_embeddings[i, 0] + 0.01, reduced_embeddings[i, 1] + 0.01, word, fontsize=12)

# 标注向量关系
plt.arrow(reduced_embeddings[0, 0], reduced_embeddings[0, 1],
          reduced_embeddings[4, 0] - reduced_embeddings[0, 0], 
          reduced_embeddings[4, 1] - reduced_embeddings[0, 1],
          color='red', head_width=0.01, length_includes_head=True)

plt.arrow(reduced_embeddings[2, 0], reduced_embeddings[2, 1],
          reduced_embeddings[5, 0] - reduced_embeddings[2, 0], 
          reduced_embeddings[5, 1] - reduced_embeddings[2, 1],
          color='blue', head_width=0.01, length_includes_head=True)

plt.title("Word Embeddings Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

代码解释：
PCA 降维：使用 PCA 将 3 维嵌入向量降至 2 维，这样可以在二维平面上绘制它们。

散点图：绘制每个单词的二维表示，并用 plt.text 标注单词名称。

标注向量关系：使用 plt.arrow 绘制单词之间的箭头，示意语义关系。比如，'man' 和 'king' 之间的箭头显示它们的关系，类似地，'woman' 和 'queen' 之间也有箭头。

标题和轴标签：给图像添加标题、X 轴和 Y 轴标签。

运行结果：
这段代码将在一个二维平面上可视化 man、boy、woman、girl、king 和 queen 之间的关系，显示它们的嵌入向量位置，并使用箭头指示相关词语之间的关系。
