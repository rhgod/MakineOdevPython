import random

import matplotlib.pyplot as plt

data = []
for i in range(50):

  features = [random.random() for j in range(450)]

  properties = [random.randint(1, 100) for j in range(5)]
  row = features + properties
  data.append(row)

for row in data:
  row[0] += 10

new_data = []
for i in range(150):
  features = [random.random() for j in range(450)]
  properties = [random.randint(1, 100) for j in range(5)]
  row = features + properties
  new_data.append(row)

data[:150] = new_data

hypothesis = "There is a positive correlation between the first feature and the fifth property"

first_feature = [row[0] for row in data]
fifth_property = [row[-1] for row in data]

plt.scatter(first_feature, fifth_property)
plt.xlabel("First Feature")
plt.ylabel("Fifth Property")
plt.title("Relationship between First Feature and Fifth Property")
plt.show()
