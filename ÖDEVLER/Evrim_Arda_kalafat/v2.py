import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

GRUP_A=450
GRUP_B=50
# Generate random numbers for the attributes of group A and group B
group_a = np.random.rand(GRUP_A, 2)
group_b = np.random.rand(GRUP_B, 2)

# Set the third attribute of group A to 1
group_a = np.hstack((group_a, np.full((GRUP_A, 1), 0.9)))
# group_a = np.hstack((group_a, (np.random.rand(grupA, 1) * (1-0.6) + 0.6 )))

# Set the third attribute of group B to random numbers between 0 and 1
group_b = np.hstack((group_b, np.random.rand(GRUP_B, 1)))

# Concatenate the groups to create the dataset
dataset = np.vstack((group_a, group_b))

# Create the labels for the dataset
labels = np.concatenate((np.ones(GRUP_A), np.zeros(GRUP_B)))

# Train a Support Vector Machine with a linear kernel on the dataset
clf = svm.SVC(kernel="linear")
clf.fit(dataset, labels)

# Print the accuracy of the classification
print("Accuracy:", clf.score(dataset, labels))

# plot
# Get the minimum and maximum values for each attribute
min_val = np.min(dataset, axis=0)
max_val = np.max(dataset, axis=0)

# Create a grid of points for each attribute
xx, yy, zz = np.meshgrid(np.linspace(min_val[0], max_val[0]),
                         np.linspace(min_val[1], max_val[1]),
                         np.linspace(min_val[2], max_val[2]))

# Use the trained classifier to predict the labels for each point on the grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

# Reshape the predictions to match the grid
Z = Z.reshape(xx.shape)

z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x-clf.coef_[0][1]*y) / clf.coef_[0][2]
tmp = np.linspace(0,1,2)
x,y = np.meshgrid(tmp,tmp)

# Create a new figure with a 3D projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the classification boundary
# ax.contourf(xx[:,:,0], yy[:,:,0], Z[:,:,0], cmap=plt.cm.Greens)

# Overlay the original dataset
ax.scatter(group_a[:, 0], group_a[:, 1], group_a[:, 2], c='r')
ax.scatter(group_b[:, 0], group_b[:, 1], group_b[:, 2], c='b')

ax.plot_surface(x, y, z(x,y), cmap=cm.ocean, alpha=0.7)

# Adjust the viewing angle
ax.view_init(elev=20, azim=-135)

plt.show()
