#!/usr/bin/env python
# coding: utf-8

# 1. Soru: 450-50 bir sınıf ve bu verilerin 5 özniteliği olacak. Değerler rasgele verilecek ve validasyon gerçekleştirilecek

# In[1]:


from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,model_selection,datasets
from scipy import stats
#Gerekli kütüphaneler eklendi

x1 = random.rand(500, 5) #Her 500 verinin 5 özniteliği rasgele olacak şekilde eklendi

y11=np.zeros(450)#İlk 450 0.sınıf olarak ayarlandı
y12=np.ones(50)#Geri kalan 50 1. sınıf olaraka ayarlandı

y1=np.append(y11,y12)


# In[2]:


plt.xlabel("0.Öznitelik")
plt.ylabel("1.Öznitelik")
plt.scatter(x1[:450, 0], x1[:450, 1],color='hotpink');
plt.scatter(x1[451:500, 0], x1[451:500, 1],color='pink');
plt.show() 
#Rasgelelik gösterildi


# In[3]:


C=1.0
clf1 = svm.LinearSVC(C=C) #Destek vektör makinesi çekideği lineer olacak şekilde ayarlandı
clf1.fit(x1,y1)

scores =model_selection.cross_val_score(clf1, x1, y1, cv=10)
print (scores)

 #Cross validation ile validasyon gerçekleştirildi


# 2. Soru: Herhangi bir özniteliğe yanlılık katılacak ve validasyonu gerçekleştirilecek

# In[4]:


x2=x1
y2=y1
for i in range(450):#Verilerin 0. özniteliğinde yanlılık eklendi
    x2[i][0]=1-(0.1*x2[i][0])

for i in range (451,500):
    x2[i][0]=0.1*x2[i][0]


# In[5]:


plt.xlabel("0.Öznitelik")
plt.ylabel("1.Öznitelik")
plt.scatter(x2[:450, 0], x2[:450, 1],color='hotpink');
plt.scatter(x2[451:500, 0], x2[451:500, 1],color='pink');
plt.show() 
#0. özniteliğin yanlılığı tabloda gösterildi


# In[6]:


clf2 = svm.LinearSVC(C=C) #Destek vektör makinesi çekideği lineer olacak şekilde ayarlandı
clf2.fit(x2,y2)

scores =model_selection.cross_val_score(clf2, x2, y2, cv=10)
print (scores)

 #Cross validation ile validasyon gerçekleştirildi


# 3. Soru: 450 ayrımını istediğimiz bir sayı yapıp doğruluk oranı tahmin edilir ve sonrada gösterilir

# 450 ayrımında 0.9 doğruluk oranı görüldü. Bunun temel nedeni verilerin bir çoğunun bir sınıfa toplanmasıdır. 270-230 bir veri ayrımı denenirse 0.52-0.56 arası bir değer beklenmektedir. Bu varsayım; 250-250 veri ayrımında doğruluk oranı beklentisinin 0.5, derste yapılan 300-200 veri ayrımının doğruluk oranının 0.6 olmasından yola çıkılmıştır.

# In[7]:


x3 = x1

y31=np.zeros(270)#İlk 270 0. sınıf olarak ayarlandı
y32=np.ones(230)#Geri kalan 230 1. sınıf olaraka ayarlandı
y3=np.append(y31,y32)


# In[8]:


clf3 = svm.LinearSVC(C=C) #Destek vektör makinesi çekideği lineer olacak şekilde ayarlandı
clf3.fit(x3,y3)

scores =model_selection.cross_val_score(clf3, x3, y3, cv=10)
print (scores)

 #Cross validation ile validasyon gerçekleştirildi


# 4. Soru: Verilerin nasıl görselleştirileceği ile ilgili öneride bulun

# Yaptığım araştırmada destek vektör makineleri numpy kütüphanesi kullanılarak görselleştirilmiştir.

# In[9]:


X1 = np.zeros(shape=(500,2))
for i in range(500): 
    X1[i][0]=x1[i][0]
    X1[i][1]=x1[i][1]

X2 = np.zeros(shape=(500,2))
for i in range(500): 
    X2[i][0]=x2[i][0]
    X2[i][1]=x2[i][1]

X3 = np.zeros(shape=(500,2))
for i in range(500): 
    X3[i][0]=x3[i][0]
    X3[i][1]=x3[i][1]


# In[10]:


clf1 = svm.LinearSVC(C=C).fit(X1, y1)
clf2 = svm.LinearSVC(C=C).fit(X2, y2)
clf3 = svm.LinearSVC(C=C).fit(X3, y3)

h=0.2

# create a mesh to plot in for each svm
x1_min, x1_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y1_min, y1_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx1, yy1 = np.meshgrid(np.arange(x1_min, x1_max, h),
                     np.arange(y1_min, y1_max, h))

x2_min, x2_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y2_min, y2_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx2, yy2 = np.meshgrid(np.arange(x2_min, x2_max, h),
                     np.arange(y2_min, y2_max, h))

x3_min, x3_max = X3[:, 0].min() - 1, X3[:, 0].max() + 1
y3_min, y3_max = X3[:, 1].min() - 1, X3[:, 1].max() + 1
xx3, yy3 = np.meshgrid(np.arange(x3_min, x3_max, h),
                     np.arange(y3_min, y3_max, h))


for i, clf in enumerate((clf1, clf2, clf3)):

    if i==0:
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z1 = clf1.predict(np.c_[xx1.ravel(), yy1.ravel()])
        Z1 = Z1.reshape(xx1.shape)
        plt.contourf(xx1, yy1, Z1, cmap=plt.cm.Pastel2, alpha=0.6)
        plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap=plt.cm.Pastel2)
        plt.xlabel("0.öznitelik")
        plt.ylabel("1.öznitelik")
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(yy1.min(), yy1.max())
        plt.xticks(())
        plt.yticks(())
        plt.title('450-50 svm veri seti')        
    elif i==1:
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z2 = clf2.predict(np.c_[xx2.ravel(), yy2.ravel()])
        Z2 = Z2.reshape(xx2.shape)
        plt.contourf(xx2, yy2, Z2, cmap=plt.cm.tab10, alpha=0.6)
        plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.tab10)
        plt.xlabel("0.öznitelik")
        plt.ylabel("1.öznitelik")
        plt.xlim(xx2.min(), xx2.max())
        plt.ylim(yy2.min(), yy2.max())
        plt.xticks(())
        plt.yticks(())
        plt.title("Yanlılık içeren svm veri seti")
    elif i==2:
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z3 = clf3.predict(np.c_[xx3.ravel(), yy3.ravel()])
        Z3= Z3.reshape(xx3.shape)
        plt.contourf(xx3, yy3, Z3, cmap=plt.cm.tab20b, alpha=0.4)
        plt.scatter(X3[:, 0], X3[:, 1], c=y3, cmap=plt.cm.tab20b)
        plt.xlabel("0.öznitelik")
        plt.ylabel("1.öznitelik")
        plt.xlim(xx3.min(), xx3.max())
        plt.ylim(yy3.min(), yy3.max())
        plt.xticks(())
        plt.yticks(())
        plt.title("270-230 svm veri seti")
       
plt.show() 