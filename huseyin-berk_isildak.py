from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets

cancer = datasets.load_breast_cancer()


cancer.target[[450, 50]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)


model = svm.SVC()


model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
