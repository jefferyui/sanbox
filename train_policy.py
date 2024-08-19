
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

# 模擬數據
data = {
    'user_id': [1, 1, 2, 2, 3, 3],
    'item_id': [10, 20, 10, 30, 20, 30],
    'interaction': [1, 1, 1, 1, 0, 1],  # 1 代表點擊，0 代表未點擊
    'feedback': [1, 0, 1, 1, 0, 0]  # 1 代表正面反饋，0 代表負面反饋
}

df = pd.DataFrame(data)

# 將 user_id 和 item_id 轉換為整數索引
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
df['item_id_encoded'] = item_encoder.fit_transform(df['item_id'])

# 特徵提取
def prepare_data(df):
    user_ids = df['user_id_encoded'].values
    item_ids = df['item_id_encoded'].values
    interactions = df['interaction'].values
    feedbacks = df['feedback'].values
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    user_one_hot = np.eye(num_users)[user_ids]
    item_one_hot = np.eye(num_items)[item_ids]
    
    features = np.hstack([user_one_hot, item_one_hot])
    
    return features, feedbacks

features, feedbacks = prepare_data(df)

# 切分訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(features, feedbacks, test_size=0.2, random_state=42)

# 定義強化學習模型
class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.output_layer(x)

def compute_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return discounted_rewards

def train_policy(model, states, actions, rewards, optimizer):
    discounted_rewards = compute_rewards(rewards)
    with tf.GradientTape() as tape:
        logits = model(states)
        loss = tf.keras.losses.sparse_categorical_crossentropy(actions, logits, from_logits=True)
        loss = tf.reduce_mean(loss * discounted_rewards)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 訓練模型
input_dim = features.shape[1]  # 特徵維度
output_dim = len(np.unique(feedbacks))  # 物品數量
policy_model = PolicyNetwork(input_dim, output_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    train_policy(policy_model, X_train_shuffled, y_train_shuffled, y_train_shuffled, optimizer)
    print(f'Epoch {epoch + 1} completed')

# 評估模型
def evaluate_model(model, features, feedbacks):
    predictions = model(features)
    predicted_actions = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_actions == feedbacks)
    return accuracy

train_accuracy = evaluate_model(policy_model, X_train, y_train)
test_accuracy = evaluate_model(policy_model, X_test, y_test)
print(f'Train accuracy: {train_accuracy:.2f}')
print(f'Test accuracy: {test_accuracy:.2f}')

# 可視化模型架構
plot_model(policy_model, to_file='policy_model.png', show_shapes=True, show_layer_names=True)

