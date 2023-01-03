import numpy as np
import inline as inline
import matplotlib
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # veri gorsellestirmesi icin
import seaborn as sns # istatiksel olarak veri gorsellestirmesi icin
import random

np.random.seed(22)
A = np.random.randn(500,5)
#random 5 sutunlu 500 tane sayi olusturulur
#print(A)
B = np.random.rand(500)
#random 500 tane sayi olusturulur
B[0:250] = 1
B[250:500]= 2
#print(B)

#Dagilimi kontrol etmek icin
plt.plot(A[0:250,1],A[0:250,2], 'r*')
plt.plot(A[250:500,1],A[250:500,2], 'b*')
plt.show()

from sklearn.model_selection import train_test_split

#Veriler egitim ve test seti olarak ayrilir
X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.2, random_state = 0)

print(X_train.shape, X_test.shape)

#Training testi hazır olduğunda, SVM Sınıflandırma Sınıfını importlayabiliriz ve eğitim setini modelimize uydurabiliriz.
# SVC sınıfı, değişken sınıflandırıcıya atanır. Burada kullanılan çekirdek, Radial Basis Function anlamına gelen “rbf” çekirdeğidir.
# Ayrıca uygulanabilen linear ve Gauss kernels gibi birkaç başka kernel da vardır.
# svc.fit() işlevi daha sonra modeli eğitmek için kullanılır.
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', C=1)#Düşük C, daha fazla aykırı değere izin verdiğimizi ve
# yüksek C daha az aykırı değere izin verdiğimizi gösterir. Bu veride C degerini arttirdikca keskinlik-dogrulugumuz azalir
svc.fit(X_train, y_train)

#bu kisimda svc.predict() islevi test setinin degerlerini tahmin etmek icin kullanilir ve degerler y_pred'e store'lanir
y_pred = svc.predict(X_test)
print("Tahmin edilen veriler:\n", y_pred)
print("\n")


#Burada, eğitilen modelin dogrulugunu-kesinligini görüyoruz ve karışıklık matrisini çiziyoruz.
#Karışıklık matrisi, Test Setinin gerçek değerleri bilindiğinde, bir sınıflandırma problemindeki doğru ve
# yanlış tahminlerin sayısını göstermek için kullanılan bir tablodur.

#True positive: Hastalığınız olduğunu düşünüyorsunuz (testin pozitif çıkacağını tahmin ettiniz) ve test pozitif çıktı.
#Yani öne sürdüğünüz hipotezin doğru olduğunu düşündünüz ve doğru çıktı.
#False positive: Hastalığınız olduğunu düşünüyorsunuz (tahmininiz pozitif) ama yaptığınız test negatif çıktı.
#False negative: Hastalığınız olmadığını düşünüyorsunuz (tahmininiz negatif) ama test yaptınız ve pozitif çıktı.
#True negative: Hastalığınız olmadığını düşünüyorsunuz (tahmininiz negatif) test yaptınız ve negatif çıktı.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
print ("Dogruluk-Kesinlik : ", accuracy_score(y_test, y_pred))
print("\n")

print('Karisiklik matrisi:\n', cm)

print('\nGercek Pozitif(TP) = ', cm[0,0])

print('\nGercek Negatif(TN) = ', cm[1,1])

print('\nTahmin Pozitif(FP) = ', cm[0,1]) #Type I error

print('\nTahmin Negatif(FN) = ', cm[1,0]) #Type II error
print("\n")

cm_matris = pd.DataFrame(data=cm, columns=['Gercek Pozitif:1', 'Gercek Negatif:0'],
                                 index=['Tahmin Pozitif:1', 'Tahmin Negatif:0'])

sns.heatmap(cm_matris, annot=True)

#Bu adımda, hem orijinal Test setinin (y_test) hem de tahmin edilen
#sonuçların (y_pred) sınıflandırılmış değerlerini karşılaştırmak için bir Pandas DataFrame oluşturulur.
pandasDataFrame = pd.DataFrame({'Gercek Veriler':y_test, 'Tahmin Edilen Veriler':y_pred})
print(pandasDataFrame)

plt.show()


B[0:450] = 1
B[450:500]= 2

plt.plot(A[0:450,1],A[0:450,2], 'r*')
plt.plot(A[450:500,1],A[450:500,2], 'b*')
plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.2, random_state = 0)

print(X_train.shape, X_test.shape)


from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', C=1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print("Tahmin edilen veriler:\n", y_pred)
print("\n")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
print ("Dogruluk-Kesinlik : ", accuracy_score(y_test, y_pred))
print("\n")

print('Karisiklik matrisi:\n', cm)

print('\nGercek Pozitif(TP) = ', cm[0,0])

print('\nGercek Negatif(TN) = ', cm[1,1])

print('\nTahmin Pozitif(FP) = ', cm[0,1]) #Type I error

print('\nTahmin Negatif(FN) = ', cm[1,0]) #Type II error
print("\n")

cm_matrix = pd.DataFrame(data=cm, columns=['Gercek Pozitif:1', 'Gercek Negatif:0'],
                                 index=['Tahmin Pozitif:1', 'Tahmin Negatif:0'])

sns.heatmap(cm_matrix, annot=True)

plt.show()

pandasDataFrame = pd.DataFrame({'Gercek Veriler':y_test, 'Tahmin Edilen Veriler':y_pred})
print(pandasDataFrame)



A = np.random.randn(500,6)
#print(A)
A[0:450, 5]=1
A[450:500, 5]=1+np.random.rand(50)*0.1

plt.plot(A[0:450,1],A[0:450,2], 'r*')
plt.plot(A[450:500,1],A[450:500,2], 'b*')
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(A, B, test_size = 0.2, random_state = 0)

print(X_train.shape, X_test.shape)


from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', C=1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print("Tahmin edilen veriler:\n", y_pred)
print("\n")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
print ("Dogruluk-Kesinlik : ", accuracy_score(y_test, y_pred))
print("\n")

print('Karisiklik matrisi:\n', cm)

print('\nGercek Pozitif(TP) = ', cm[0,0])

print('\nGercek Negatif(TN) = ', cm[1,1])

print('\nTahmin Pozitif(FP) = ', cm[0,1]) #Type I error

print('\nTahmin Negatif(FN) = ', cm[1,0]) #Type II error
print("\n")

cm_matrix = pd.DataFrame(data=cm, columns=['Gercek Pozitif:1', 'Gercek Negatif:0'],
                                 index=['Tahmin Pozitif:1', 'Tahmin Negatif:0'])

sns.heatmap(cm_matrix, annot=True)

plt.show()

pandasDataFrame = pd.DataFrame({'Gercek Veriler':y_test, 'Tahmin Edilen Veriler':y_pred})
print(pandasDataFrame)