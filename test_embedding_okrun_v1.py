import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Drop rows with missing values
titanic = titanic.dropna(subset=['sex', 'embarked', 'age', 'fare'])

# Encode categorical features
label_encoders = {}
for column in ['sex', 'embarked']:
    le = LabelEncoder()
    titanic[column] = le.fit_transform(titanic[column])
    label_encoders[column] = le

# Confirm the unique values and their encoded values
print("Sex encoding:", label_encoders['sex'].classes_)
print("Embarked encoding:", label_encoders['embarked'].classes_)

# Normalize numerical features
scaler = StandardScaler()
titanic[['age', 'fare']] = scaler.fit_transform(titanic[['age', 'fare']])

# Features and target
features = ['sex', 'embarked', 'age', 'fare']
X = titanic[features].values
y = titanic['survived'].values

from tensorflow.keras import layers, models
import tensorflow as tf
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def build_model():
    inputs ={}
    # Define feature column parameters
    num_categories_sex = len(label_encoders['sex'].classes_)  # Should be 2 for 'sex' (male, female)
    num_categories_embarked = len(label_encoders['embarked'].classes_)  # Should be 3 for 'embarked' (C, Q, S)
    embedding_dim = 4
    
    # Define input layers for each feature
    inputs['sex'] = tf.keras.Input(shape=(1,), name='sex', dtype=tf.int32)
    inputs['embarked'] = tf.keras.Input(shape=(1,), name='embarked', dtype=tf.int32)
    inputs['age'] = tf.keras.Input(shape=(1,), name='age')
    inputs['fare'] = tf.keras.Input(shape=(1,), name='fare')
    
    # Create embedding layers for categorical features
    embedding_sex = tf.keras.layers.Embedding(input_dim=num_categories_sex, output_dim=embedding_dim)(inputs['sex'])
    embedding_sex = tf.keras.layers.Flatten()(embedding_sex)
    
    embedding_embarked = tf.keras.layers.Embedding(input_dim=num_categories_embarked, output_dim=embedding_dim)(inputs['embarked'])
    embedding_embarked = tf.keras.layers.Flatten()(embedding_embarked)
    
    # Concatenate all features
    concatenated_features = tf.keras.layers.Concatenate()([embedding_sex, embedding_embarked, inputs['age'], inputs['fare']])
    
    # Build the rest of the model
    x = tf.keras.layers.Dense(16, activation='relu')(concatenated_features)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=output)
    
    return model

# 构建并编译模型
# 构建并编译模型
model = build_model()
# Create the model
# model = tf.keras.Model(inputs=[input_sex, input_embarked, input_age, input_fare], outputs=output)

# # Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare data using tf.data.Dataset
def create_dataset(X, y, batch_size=32, shuffle_buffer_size=1000):
    dataset = tf.data.Dataset.from_tensor_slices(({
        'sex': X[:, 0],
        'embarked': X[:, 1],
        'age': X[:, 2],
        'fare': X[:, 3]
    }, y))
    
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    return dataset

# Create datasets
train_dataset = create_dataset(X_train, y_train)
test_dataset = create_dataset(X_test, y_test)

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Predict with the model
predictions = model.predict(test_dataset)
print("Model Predictions:", predictions[:5])
# # Prepare the data for training
# train_data = {
#     'sex': X_train[:, 0],
#     'embarked': X_train[:, 1],
#     'age': X_train[:, 2],
#     'fare': X_train[:, 3]
# }

# test_data = {
#     'sex': X_test[:, 0],
#     'embarked': X_test[:, 1],
#     'age': X_test[:, 2],
#     'fare': X_test[:, 3]
# }
# model.fit(train_data, y_train, epochs=10, batch_size=32, validation_split=0.1)

# # Evaluate the model
# loss, accuracy = model.evaluate(test_data, y_test)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy:.4f}")

# # Predict with the model
# predictions = model.predict(test_data)
# print("Model Predictions:", predictions[:5])
