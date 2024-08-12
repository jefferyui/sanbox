
# use titanic dataset do again

import pandas as pd
import seaborn as sns

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Preview the data
df.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Drop rows with missing target or important features
df = df.dropna(subset=['age', 'fare', 'embarked', 'sex', 'pclass', 'survived'])

# Select features and target
features = ['pclass', 'sex', 'age', 'fare', 'embarked']
target = 'survived'

X = df[features]
y = df[target]

# Convert categorical columns to strings for processing
X['sex'] = X['sex'].astype(str)
X['embarked'] = X['embarked'].astype(str)
X['pclass'] = X['pclass'].astype(str)  
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate numerical and categorical features
num_features = ['age', 'fare']
cat_features = ['pclass', 'sex', 'embarked']

# Normalize numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_features])
X_test_num = scaler.transform(X_test[num_features])

# Prepare categorical features using StringLookup and IntegerLookup
lookup_layers = {}
for feature in cat_features:
    lookup_layer = tf.keras.layers.StringLookup(output_mode='int')
    lookup_layer.adapt(X_train[feature])
    lookup_layers[feature] = lookup_layer

# Convert categorical features to integer indices
X_train_cat = {feature: lookup_layers[feature](X_train[feature].values) for feature in cat_features}
X_test_cat = {feature: lookup_layers[feature](X_test[feature].values) for feature in cat_features}

from tensorflow.keras import layers, models

def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    # Multi-Head Self Attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    # Feed Forward Layer
    ff_output = layers.Dense(ff_dim, activation='relu')(attention_output)
    ff_output = layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    transformer_output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
    return transformer_output

def build_model(num_features, cat_vocab_sizes, embedding_dim=4, num_heads=2, ff_dim=32):
    # Numerical input
    input_num = layers.Input(shape=(num_features,))
    x_num = layers.Dense(32, activation='relu')(input_num)

    # Categorical input
    cat_inputs = []
    cat_embeddings = []
    for vocab_size in cat_vocab_sizes:
        input_cat = layers.Input(shape=(1,))
        cat_inputs.append(input_cat)
        x_cat = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_cat)
        x_cat = layers.Flatten()(x_cat)
        cat_embeddings.append(x_cat)

    # Concatenate all inputs
    x = layers.Concatenate()([x_num] + cat_embeddings)
    x = layers.Dense(64, activation='relu')(x)

    # Add Transformer blocks
    x = layers.Reshape((-1, x.shape[-1]))(x)  # Reshape for the Transformer
    for _ in range(2):  # Example: 2 Transformer blocks
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)

    x = layers.Flatten()(x)  # Flatten the output of the Transformer

    # Add final hidden layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Output layer
    output = layers.Dense(1, activation='sigmoid')(x)

    # Define the model
    model = models.Model(inputs=[input_num] + cat_inputs, outputs=output)

    return model

# Vocabulary sizes for categorical features
cat_vocab_sizes = [len(lookup_layers[feature].get_vocabulary()) for feature in cat_features]

# Build the model
model = build_model(num_features=len(num_features), cat_vocab_sizes=cat_vocab_sizes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()
# Prepare inputs for training
X_train_inputs = [X_train_num] + [X_train_cat[feature] for feature in cat_features]
X_test_inputs = [X_test_num] + [X_test_cat[feature] for feature in cat_features]

# Train the model
model.fit(X_train_inputs, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_inputs, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
# UNIMPLEMENTED: Cast int64 to string is not supported for pclass

# add model predict

# Make predictions on the test set
predictions = model.predict(X_test_inputs)

# Since the output is a probability, you may want to threshold it to get binary predictions
predictions_binary = (predictions > 0.5).astype(int)

# Display the predictions
for i in range(5):
    print(f"Passenger {i+1}: Predicted Survival = {predictions_binary[i][0]}, Actual Survival = {y_test.iloc[i]}")

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions_binary)
print(f"Prediction Accuracy: {accuracy}")
