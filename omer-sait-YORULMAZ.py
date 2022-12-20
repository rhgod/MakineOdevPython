
data = []
with open('data.csv', 'r') as f:
  reader = csv.reader(f)
  for row in reader:
    data.append(row)

x1 = [int(row[0]) for row in data]
x2 = [int(row[1]) for row in data]
x3 = [int(row[2]) for row in data]
x4 = [int(row[3]) for row in data]
x5 = [int(row[4]) for row in data]

error = 50
x3[450] += error

x1[450] = 0
x2[450] = 0
x4[450] = 0
x5[450] = 0

b1, a1 = np.polyfit(x1, x3, 1)
b2, a2 = np.polyfit(x2, x3, 1)
b3, a3 = np.polyfit(x4, x3, 1)
b4, a4 = np.polyfit(x5, x3, 1)

hypothesis1 = [a1 + b1 * xi for xi in x1]
hypothesis2 = [a2 + b2 * xi for xi in x2]
hypothesis3 = [a3 + b3 * xi for xi in x4]
hypothesis4 = [a4 + b4 * xi for xi in x5]

plt.scatter(x1, x3)
plt.plot(x1, hypothesis1, 'r')
plt.xlabel('x1')
plt.ylabel('x3')
plt.title('Linear regression with error')
plt.show()

plt.scatter(x2, x3)
plt.plot(x2, hypothesis2, 'r')
plt.xlabel('x2')
plt.ylabel('x3')
plt.title('Linear regression with error')
plt.show()

plt.scatter(x4, x3)
plt.plot(x4, hypothesis3, 'r')
plt.xlabel('x4')
plt.ylabel('x3')
plt.title('Linear regression with error')
plt.show()

plt.scatter(x5, x3)
plt.plot(x5, hypothesis4, 'r')
plt.xlabel('x5')
plt.ylabel('x3')
plt.title('Linear regression with error')
plt.show()