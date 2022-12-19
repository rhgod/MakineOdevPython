import csv
import numpy as np
import matplotlib.pyplot as plt
with open('veriler.csv', 'r') as f:
 reader = csv.reader(f)
 attributes = []
 data = []

 attributes = next(reader)

 for row in reader:
 data.append(row)
data = np.array(data)
X = data[:, :5]
error = np.random.normal(0, 0.1, 50)
X[:, 3] += error
X = X[:50]
plt.scatter(X[:, 0], X[:, 1], c=X[:, 2])
plt.show()