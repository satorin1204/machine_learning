import numpy as np
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
logistic = LogisticRegression()
svc = svm.SVC(kernel='linear')
# Logistic Regression
scores = cross_val_score(logistic, iris.data, iris.target, cv=5)
print("logistic regression", np.mean(scores))

# SVM(kernel:rbf)
scores = cross_val_score(svc, iris.data, iris.target, cv=5)
print("svm(linear): ", np.mean(scores))

scores = cross_val_score(KNeighborsClassifier(), iris.data, iris.target, cv=5)
print("knn: ", np.mean(scores))
