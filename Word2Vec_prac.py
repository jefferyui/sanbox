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

