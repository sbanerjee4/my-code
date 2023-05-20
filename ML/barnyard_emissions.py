# -*- coding: utf-8 -*-

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas_profiling as pp
import matplotlib.pyplot as plt
from matplotlib import pyplot
from google.colab import files
import seaborn as sns
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix

# https://data.nal.usda.gov/dataset/data-gas-emissions-dairy-barnyards/resource/2b611df0-07b4-4e0b-a490-b5e045369b82#{view-graph:{graphOptions:{hooks:{processOffset:{},bindEvents:{}}}},graphOptions:{hooks:{processOffset:{},bindEvents:{}}}}
file_name = 'BYD_Intake.csv'
df = pd.read_csv(file_name)

# check columns with many missing values
df.info()

# create year and month columns
df['Year'] = df.Date.apply(lambda x: int(x[:4]))
df['Month'] = df.Date.apply(lambda x: int(x[5:7]))

# delete date column (converted into year/month)
# delete feed intake lb column (feed intake lb * 2.2 = feed intake kg)
# delete C_content, NDF_content, Notes columns (too many NA values)
drop_columns = ['Date', 'Feed_Intake_lb', 'C_content', 'NDF_content', 'Notes']
df = df.drop(columns=drop_columns, axis=1)
df.info()

# https://www.analyticsvidhya.com/blog/2021/05/dealing-with-missing-values-in-python-a-complete-guide/
df.info()
print(df.isnull().sum())

updated_df = df
updated_df['Refusals']=updated_df['Refusals'].fillna(updated_df['Refusals'].mean())
updated_df['N_content']=updated_df['N_content'].fillna(updated_df['N_content'].mean())
updated_df['N_intake']=updated_df['N_intake'].fillna(updated_df['N_intake'].mean())

updated_df.info()
print(df.isnull().sum())

array = np.array(updated_df)

# https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
scaler = preprocessing.StandardScaler().fit(array)
scaled_array = scaler.transform(array)
scaled_array.mean(axis=0)
scaled_array.std(axis=0)
array = scaled_array

# https://scikit-learn.org/stable/modules/preprocessing.html#normalization
normalizer = preprocessing.Normalizer().fit(array)  # fit does nothing
normalized_array = normalizer.transform(array)
array = normalized_array

df = pd.DataFrame(df, columns=['Barnyard', 'Total_mixed_rations', 'Refusals', 'Feed_intake_kg',
       'Dry_matter', 'N_content', 'P_content', 'N_intake', 'P_intake', 'Year', 'Month'])

sns.set()
for col in ((df.columns)[:-2]):
  plt.figure()
  heatmap_df = pd.pivot_table(df, values=col, index=['Year'], columns=['Month'], aggfunc=np.mean, fill_value=0)
  ax = sns.heatmap(heatmap_df, annot=True)
  ax.set_title(col)
  plt.savefig(col.join('.png'))

# https://towardsdatascience.com/a-beginners-guide-to-data-analysis-in-python-188706df5447

print(df.columns)
for col in df.columns:
  (df[col]).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
  pyplot.show()
  (df[col]).plot(kind='hist', subplots=True, layout=(2,2), sharex=False, sharey=False)
  pyplot.show()

scatter_matrix(df, figsize=(15,15), alpha=0.2)
pyplot.show()

tup = list()
for col in df.columns:
  for col2 in df.columns:
    a, b = df[col], df[col2]
    correlation = a.corr(b)
    tup += [(col, col2, correlation)]

tup.sort(key=lambda x:x[2])
for t in tup:
  print(t)

X = df.drop(['Month'],axis=1).values   # independent features
y = df['Month'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new)

from sklearn import metrics

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Gaussian NB
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# SVM
from sklearn.svm import SVC
clf = SVC().fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# SGDC
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=200).fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Ensemble
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
clf = bagging.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Neural Network
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Confusion Matrix
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)