# -*- coding: utf-8 -*-

# import necessary libraries
import pandas as pd
import numpy as np
import statistics

# read data from csv file
data = pd.read_csv('HCV-Egy-Data.csv')

# separates data into features and target
X_data = data.drop('Baselinehistological staging', axis = 1)
y_data = data['Baselinehistological staging']

import seaborn as sns
import matplotlib.pyplot as plt

# generate heatmap for all features in relation to each other
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot = False)
plt.show()

sns.set()
# generate heatmaps for each individual feature in relation to the target
for col in ((data.columns)[:-1]):
  plt.figure()
  heatmap = pd.pivot_table(data, values = col, columns = ['Baselinehistological staging'], aggfunc = np.mean, fill_value = 0)
  ax = sns.heatmap(heatmap, annot = True)
  ax.set_title(col)

# generate graph for all features colletively in relation to the target
plt.plot(X_data, y_data, color = 'green')
plt.xlabel('Features')
plt.ylabel('Baselinehistological staging')
plt.axes().set_facecolor('black')
plt.show()

# generate graphs for each individual feature in relation to the target
for col in ((data.columns)[:-1]):
  plt.plot(data[col], y_data, color = 'blue')
  plt.xlabel(col)
  plt.ylabel('Baselinehistological staging')
  plt.axes().set_facecolor('black')
  plt.show()

from sklearn.preprocessing import StandardScaler

# normalize feature values by scaling them down
scaler = StandardScaler()
scaler.fit(X_data)
scaler.transform(X_data)

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.20)

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracies = []

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# create model and run with test data
clf = DecisionTreeClassifier().fit(X_train,y_train)
y_pred = clf.predict(X_test)
ac = metrics.accuracy_score(y_test, y_pred)
print("Decision Tree")
print("Accuracy:", ac)
accuracies.append(ac)

# generate heatmap from confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot = True, cmap = 'Blues')
ax.set_title('Decision Tree Confusion Matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
plt.show()

# Neural Network
from sklearn.neural_network import MLPClassifier

# create model and run with test data
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ac = metrics.accuracy_score(y_test, y_pred)
print("Neural Network")
print("Accuracy:", ac)
accuracies.append(ac)

# generate heatmap from confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot = True, cmap = 'Blues')
ax.set_title('Neural Network Confusion Matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
plt.show()

# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# create model and run with test data
clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ac = metrics.accuracy_score(y_test, y_pred)
print("K Nearest Neighbors")
print("Accuracy:", ac)
accuracies.append(ac)

# generate heatmap from confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot = True, cmap = 'Blues')
ax.set_title('KNN Confusion Matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
plt.show()

# Bagging Classifier
from sklearn.ensemble import BaggingClassifier

# create model and run with test data
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
clf = bagging.fit(X_train,y_train)
y_pred = clf.predict(X_test)
ac = metrics.accuracy_score(y_test, y_pred)
print("Bagging Classifier")
print("Accuracy:", ac)
accuracies.append(ac)

# generate heatmap from confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot = True, cmap = 'Blues')
ax.set_title('Bagging Classifier Confusion Matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
plt.show()

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

# create model and run with test data
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=200).fit(X_train,y_train)
y_pred = clf.predict(X_test)
ac = metrics.accuracy_score(y_test, y_pred)
print("Stochastic Gradient Descent")
print("Accuracy:", ac)
accuracies.append(ac)

# generate heatmap from confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot = True, cmap = 'Blues')
ax.set_title('SGD Confusion Matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
plt.show()

# Support Vector Machine
from sklearn.svm import SVC

# create model and run with test data
clf = SVC().fit(X_train,y_train)
y_pred = clf.predict(X_test)
ac = metrics.accuracy_score(y_test, y_pred)
print("Support Vector Machine")
print("Accuracy:", ac)
accuracies.append(ac)

# generate heatmap from confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot = True, cmap = 'Blues')
ax.set_title('SVM Confusion Matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
plt.show()

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# create model and run with test data
clf = GaussianNB().fit(X_train,y_train)
y_pred = clf.predict(X_test)
ac = metrics.accuracy_score(y_test, y_pred)
print("Gaussian Naive Bayes")
print("Accuracy:", ac)
accuracies.append(ac)

# generate heatmap from confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot = True, cmap = 'Blues')
ax.set_title('GNB Confusion Matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');
plt.show()

# statistical values of accuracies
print("Accuracies")
print("Mean:\t", statistics.mean(accuracies))
print("Median:\t", statistics.median(accuracies))
print("Minimum:", min(accuracies))
print("Maximum:", max(accuracies))