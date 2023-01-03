import random
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

# Load data
(X, y), (X_test, y_test) = mnist.load_data()

# Shuffle data
random.seed(42)
indices = list(range(len(X)))
random.shuffle(indices)

# Split train and test
train_indices = indices[:5000]
test_indices = indices[5000:6000]
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]


# Reshape data
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# Normalize data
X_train = X_train / 255
X_test = X_test / 255

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,), activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Predict
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Evaluate
print(accuracy_score(y_test, y_pred))
