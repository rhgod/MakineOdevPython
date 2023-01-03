# Load MNIST dataset
import random
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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

# Plot example image in bigger size (for better visualization)
for i in range(5):
    random_int = random.randint(0, 5000)
    cv2.imshow("Example image", cv2.resize(X_train[random_int].reshape(28, 28), (200, 200)))
    cv2.waitKey(0)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
print(accuracy_score(y_test, y_pred))

