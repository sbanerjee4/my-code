# -*- coding: utf-8 -*-

import pandas as pd
from pandas import read_csv
import numpy as np
from numpy import random
from numpy import nan
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
train_df = read_csv('training.csv')
test_df = read_csv('testing.csv')
print(train_df)

def calculate_median(l):
    l = sorted(l)
    l_len = len(l)
    if l_len < 1:
        return None
    if l_len % 2 == 0 :
        return (l[(int)((l_len - 1) / 2)] + l[(int)((l_len + 1) / 2)]) / 2.0
    else:
        return l[(int)((l_len - 1) / 2)]

# fill with medians
cols_fillna = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col in cols_fillna:
  train_df[col].fillna(train_df[col].mean(), inplace = True)

# format gender column
train_df["Sex"].fillna(random.randint(0, 1), inplace = True)
train_df["Sex"] = train_df["Sex"].replace('male', 0)
train_df["Sex"] = train_df["Sex"].replace('female', 1)

# format cabin column
train_df["Letters"] = nan
train_df["Numbers"] = nan

for i in range(len(train_df)):
  list_letters = list()
  list_numbers = list()
  if train_df.loc[i, "Cabin"] is not nan:
    list_orig = train_df.loc[i, "Cabin"].split(" ")
    for s in list_orig:
      list_letters.append(ord(list(s[0])[0]))
      if len(s) > 1: list_numbers.append(int(s[1:]))
    if len(list_letters) > 0: train_df.loc[i, "Letters"] = sum(list_letters) / len(list_letters)
    if len(list_numbers) > 0: train_df.loc[i, "Numbers"] = sum(list_numbers) / len(list_numbers)

train_df["Letters"].fillna(train_df["Letters"].mean(), inplace = True)
train_df["Numbers"].fillna(train_df["Numbers"].mean(), inplace = True)

# format embarked column
train_df["Embarked"].fillna(random.randint(0, 2), inplace = True)
train_df["Embarked"] = train_df["Embarked"].replace('C', 0)
train_df["Embarked"] = train_df["Embarked"].replace('Q', 1)
train_df["Embarked"] = train_df["Embarked"].replace('S', 2)

# fill with medians
cols_fillna = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col in cols_fillna:
  test_df[col].fillna(test_df[col].mean(), inplace = True)

# format gender column
test_df["Sex"].fillna(random.randint(0, 1), inplace = True)
test_df["Sex"] = test_df["Sex"].replace('male', 0)
test_df["Sex"] = test_df["Sex"].replace('female', 1)

# format cabin column
test_df["Letters"] = nan
test_df["Numbers"] = nan

for i in range(len(test_df)):
  list_letters = list()
  list_numbers = list()
  if test_df.loc[i, "Cabin"] is not nan:
    list_orig = test_df.loc[i, "Cabin"].split(" ")
    for s in list_orig:
      list_letters.append(ord(list(s[0])[0]))
      if len(s) > 1: list_numbers.append(int(s[1:]))
    if len(list_letters) > 0: test_df.loc[i, "Letters"] = sum(list_letters) / len(list_letters)
    if len(list_numbers) > 0: test_df.loc[i, "Numbers"] = sum(list_numbers) / len(list_numbers)

test_df["Letters"].fillna(test_df["Letters"].mean(), inplace = True)
test_df["Numbers"].fillna(test_df["Numbers"].mean(), inplace = True)

# format embarked column
test_df["Embarked"].fillna(random.randint(0, 2), inplace = True)
test_df["Embarked"] = test_df["Embarked"].replace('C', 0)
test_df["Embarked"] = test_df["Embarked"].replace('Q', 1)
test_df["Embarked"] = test_df["Embarked"].replace('S', 2)

train_df.drop("Cabin", 1, inplace=True)
test_df.drop("Cabin", 1, inplace=True)

# drop_columns = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Letters", "Numbers", "Embarked"]
drop_columns = ["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Letters", "Numbers", "Embarked"]
for col in drop_columns:
  train_df.drop(col, 1, inplace=True)
  test_df.drop(col, 1, inplace=True)

train_df = train_df.sample(frac=1)
print(train_df)

X_train = train_df.loc[:, train_df.columns != "Survived"]
y_train = train_df.loc[:, "Survived"]

X_test = test_df.loc[:, test_df.columns != "Survived"]
#X_x = train_df.loc[430:, train_df.columns != "Survived"]
#X_y = train_df.loc[430:, "Survived"]

#### Scale all features so that feature values do not exert undue weights
# https://stackoverflow.com/questions/13324071/scaling-data-in-scikit-learn-svm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print ("Scaled X_train")
print(X_train)
print ("Scaled X_test")
X_test = scaler.transform(X_test)
print(X_test)

#### Use SVM Classifier directly
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
#### End of SVM Classifier

##### Perform Grid Search with SVM 
# https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# defining parameter range and grid search with SVC
from sklearn.svm import SVC
estimator = SVC()
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#             'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001],
#            'degree': [1, 2, 3, 4, 5],
#            'kernel': ['linear','poly','rbf','sigmoid']}

param_grid = {'C': [0.001, 0.01, 0.1],
             'gamma': [10],
             'degree': [3],
             'kernel': ['poly']}
# defining parameter range and grid search with RFC
# https://stackoverflow.com/questions/30102973/how-to-get-best-estimator-on-gridsearchcv-random-forest-classifier-scikit

#from sklearn.ensemble import RandomForestClassifier
# estimator = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
#param_grid = { 
#    'n_estimators': [50, 100, 200, 700, 1000],
#    'max_features': ['auto', 'sqrt', 'log2']
#}

# Perform Grid Search with the estimator and parameters
grid = GridSearchCV(estimator, param_grid, refit = True, verbose = 3)

# fitting the model for grid search
grid.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = grid.predict(X_test)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

# print best parameter after tuning
print(grid.best_params_)

# print classification report
# print(classification_report(y_test, y_pred))

#### End of Perform Grid Search with SVM

# print("Accuracy:",metrics.accuracy_score(X_y, y_pred))

# "PassengerId", Accuracy: 0.5854922279792746
# "Pclass", Accuracy: 0.7253886010362695
# "Name", N/A
# "Sex", Accuracy: 0.7668393782383419
# "Age", Accuracy: 0.5854922279792746
# "SibSp", Accuracy: 0.5854922279792746
# "Parch", Accuracy: 0.5854922279792746
# "Ticket", N/A
# "Fare", Accuracy: 0.6113989637305699
# "Letters", Accuracy: 0.5854922279792746
# "Numbers", Accuracy: 0.5854922279792746
# "Letters" and "Numbers", Accuracy: 0.5854922279792746
# "Embarked", Accuracy: 0.5803108808290155
# "Everything", Accuracy: 0.7668393782383419
# "Pclss", "Sex", and "Fare", Accuracy: 0.7668393782383419

for i in y_pred: print(i)