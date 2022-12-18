import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate a random dataset with 2 attributes and 500 elements
np.random.seed(0)
X = np.random.random((500, 2))

# Divide the dataset into group A with the first 450 elements and group B with the remaining 50 elements
y = np.concatenate([np.zeros(450), np.ones(50)])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a SVM classifier with linear kernel on the training set
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
accuracy = clf.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Plot the classification boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')
plt.show()
