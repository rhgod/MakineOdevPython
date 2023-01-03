# Load MNIST dataset
import random
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from keras.datasets import mnist
import cv2

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

# Train svm model
svm = SVC()
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Evaluate
print(accuracy_score(y_test, y_pred))
