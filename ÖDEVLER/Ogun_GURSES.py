import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

np.random.seed(20)
X = np.random.randn(500, 5)
Y = np.random.rand(500)
print(" \n kafama göre bölerek oluşturduğum iki kümenin dağılımı")
Y[0:300] = 1
Y[300:500] = 2
plt.plot(X[0:300, 1], X[0:300, 2], 'cx')
plt.plot(X[300:500, 1], X[300:500, 2], 'yH')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
svc = SVC(kernel='rbf', C=1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print("Tahmin verileri:\n", y_pred)
print("\n")
cm = confusion_matrix(y_test, y_pred)

print("Doğruluk oranı: ", accuracy_score(y_test, y_pred))
print("\n")
print('Matris:\n', cm)
print('\nGerçek Pozitif(GP) = ', cm[0, 1])
print('\nGerçek Negaitf(GN) = ', cm[1, 1])
print('\nTahmin Pozitif(TP) = ', cm[0, 1])
print('\nTahmin Negatif(TN) = ', cm[1, 0])
print("\n")

cm_matrix = pd.DataFrame(data=cm,
                         columns=['Gercek Pozitif:1', 'Gercek Negatif:0'],
                         index=['Tahmin Pozitif:1', 'Tahmin Negatif:0'])

sns.heatmap(cm_matrix, annot=True)

plt.show()

Y[0:450] = 1
Y[450:500] = 2

plt.plot(X[0:450, 1], X[0:450, 2], 'cx')
plt.plot(X[450:500, 1], X[450:500, 2], 'yH')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)


svc = SVC(kernel ='rbf', C=1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print("Tahmin verileri:\n", y_pred)
print("\n")


cm = confusion_matrix(y_test, y_pred)

print("Doğruluk Oranı: ", accuracy_score(y_test, y_pred))
print("\n")

print('Matris:\n', cm)
print('\nGerçek Pozitif(GP) = ', cm[0, 0])
print('\nGerçek Negatif(GN) = ', cm[1, 1])
print('\nTahmin Pozitif(TP) = ', cm[0, 1])
print('\nTahmin Negatif(TN) = ', cm[1, 0])
print("\n")

cm_matrix = pd.DataFrame(data=cm, columns=['Gercek Pozitif:1', 'Gercek Negatif:0'],
                                 index=['Tahmin Pozitif:1', 'Tahmin Negatif:0'])

sns.heatmap(cm_matrix, annot=True)

plt.show()
X = np.random.randn(500, 6)

X[0:450, 5] = 1
X[450:500, 5] = 1 + np.random.rand(50) * 0.2
plt.plot(X[0:450, 1], X[0:450, 2], 'cx')
plt.plot(X[450:500, 1], X[450:500, 2], 'yH')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
svc = SVC(kernel ='rbf', C=1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print("Tahmin edilen veriler:\n", y_pred)
print("\n")
cm = confusion_matrix(y_test, y_pred)
print("Dogruluk-Kesinlik : ", accuracy_score(y_test, y_pred))
print("\n")
print('Matris:\n', cm)
print('\nGerçek Pozitif(GP) = ', cm[0, 0])
print('\nGerçek Negatif(GN) = ', cm[1, 1])
print('\nTahmin Pozitif(TP) = ', cm[0, 1])
print('\nTahmin Negatif(TN) = ', cm[1, 0])
print("\n")
cm_matrix = pd.DataFrame(data=cm, columns=['Gercek Pozitif:1', 'Gercek Negatif:0'],
                         index=['Tahmin Pozitif:1', 'Tahmin Negatif:0'])
sns.heatmap(cm_matrix, annot=True)
plt.show()
