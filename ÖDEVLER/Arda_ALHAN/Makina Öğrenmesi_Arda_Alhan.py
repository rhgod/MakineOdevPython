from sklearn import svm
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.svm import SVC

eğitim= int(651)
test=int(801)

# Eğitim ve test verilerini oku
datasets = pd.read_csv('Pokemon.csv')
train_x = datasets.iloc[1:eğitim, [1,2,3,4,5]].values
train_y = datasets.iloc[1:eğitim, 6].values
test_x = datasets.iloc[eğitim:test, [1,2,3,4,5]].values
test_y = datasets.iloc[eğitim:test, 6].values

# SVM modelini oluştur
model = svm.SVC()

# Eğitim verilerini kullanarak modeli eğit
model.fit(train_x, train_y)

# Test verilerini kullanarak modelin doğruluğunu ölç
accuracy = model.score(test_x, test_y)

# Doğruluğu yazdır
print("Accuracy:", accuracy)
