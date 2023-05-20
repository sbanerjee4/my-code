#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Copyright Sanchali Banerjee


# In[8]:


import pandas as pd
from pandas import read_csv
import numpy as np

# from sklearn.preprocessing import label_binarize

from sklearn import metrics
from sklearn.metrics import accuracy_score
#from sklearn.metrics import roc_curve, auc
#from sklearn.metrics import roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
# import warnings filter
from warnings import simplefilter


# In[9]:


# Program wide settings
# Ignore warnings
# warnings.filterwarnings("ignore")
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Random seed for reproducibility
seed = 10
np.random.seed(seed)


# In[ ]:


# Hyperparameters
'''
hp_epoch
hp_learning_rate
hp_opt
hp_batch_size
hp_scaler
hp_hidden_layer
hp_hidden_act_fn
hp_output_act_fn
hp_dropout
'''


# In[10]:


# Data Ingest
df = read_csv('~/Downloads/LandType.csv')
# Print first 10 samples
print(df.head(10))


# In[11]:


# Reshuffle the dataset
df = df.sample(frac = 1).reset_index(drop = True)


# In[12]:


# Only use bands and land types
'''
X_data = df.drop(['ForestAggreg', df.columns[0]], axis = 1)
y_data = df['ForestAggreg']
'''
X_data = df.drop(['X', 'Y', 'Date','Land_Type'], axis = 1)
y_data = df['Land_Type']


# In[13]:


X_data


# In[14]:


y_data


# In[15]:


# Unique Classes
unique_classes = y_data.unique()
class_num = len(unique_classes)
class_num


# In[16]:


unique_classes


# In[17]:


# https://sparkbyexamples.com/pandas/pandas-groupby-count-examples/
# This gives a sense if the labels have adequate representation because if they don't,
# the training will be inadequate
print (df.groupby(['Land_Type']).size())


# In[18]:


pip install eli5


# In[19]:


# Feature Selection
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2

# clf = RandomForestClassifier(n_estimators = 50)
clf = PermutationImportance(RandomForestClassifier(), cv = 5)
clf = clf.fit(X_data, y_data)
selector = SelectFromModel(clf, threshold = 0.0001, prefit = True)

# Another way to do feature selection using SelectKBest
# selector = SelectKBest(score_func=f_regression, k=5)

# finding selected column names
feature_idx = selector.get_support(indices=True)
feature_names = X_data.columns[feature_idx]

# Print weights of features
from eli5 import show_weights
show_weights(clf, feature_names = X_data.columns.tolist(), top=20)


# In[21]:


# X_data can be transformed to include the selected features 
X_data = X_data.loc[:, feature_names]
# X_data = selector.transform(X_data)
print (X_data.shape)
print(feature_names)


# In[22]:


X_data


# In[23]:


# Normalize features within range 0 (minimum) and 1 (maximum)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_data = scaler.fit_transform(X_data)
X_data = pd.DataFrame(X_data)


# In[24]:


X_data


# In[25]:


# Normalize feature values. This can leak data into test. StandardScaler preferred
from sklearn import preprocessing
X_data = preprocessing.scale(X_data)


# In[26]:


X_data


# In[27]:


# Normalize feature values by scaling them down
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_data)
X_data = scaler.transform(X_data)


# In[28]:


X_data


# In[29]:


# Convert target Y to one hot encoded Y for Neural Network
y_data = pd.get_dummies(y_data)


# In[30]:


y_data


# In[31]:


# Train-test split if needed
from sklearn.model_selection import train_test_split
test_size = 0.3
# train_data, test_data = train_test_split(df, test_size=0.2045, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=42)
print ('X_train, X_test, y_train, y_test shapes: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[32]:


X_train


# In[33]:


y_train


# In[34]:


X_test


# In[35]:


y_test


# In[36]:


# Define cross validation
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import cross_val_predict

k = 5
kfold = KFold(n_splits = k, shuffle = True, random_state = seed)
# StratifiedKFold does not seem to be working with one-hot-encoding
# kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = seed)
cv = kfold


# In[37]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.optimizers import SGD, Adam
from keras.wrappers.scikit_learn import KerasClassifier


# In[38]:


X_data.shape


# In[39]:


y_data.shape


# In[40]:


X_data


# In[41]:


# convert integers to dummy variables (i.e. one hot encoded)
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(y_data)
dummy_y


# In[42]:


y_data


# In[43]:


# Calculate input dimensions for tensorflow
input_dim = np.shape(X_data)[1]
input_dim


# In[44]:


# First define baseline model. Then use it in Keras Classifier for the training
dropout = 0.2
print('Input Dim: ', input_dim)

def baseline_model():
    # Create model here
    model = Sequential()
    model.add(Dense(units=input_dim, activation='sigmoid', input_dim=input_dim, kernel_initializer = 'he_normal'))
    model.add(Dense(128, activation='relu', kernel_initializer = 'he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='sigmoid', kernel_initializer = 'he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='sigmoid', kernel_initializer = 'he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='sigmoid', kernel_initializer = 'he_normal'))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='sigmoid', kernel_initializer = 'he_normal'))
#    model.add(Dropout(dropout))
#    model.add(Dense(64, activation='relu', kernel_initializer = 'he_normal'))
#    model.add(Dropout(dropout))
#    model.add(Dense(32, activation='relu', kernel_initializer = 'he_normal'))
#    model.add(Dropout(dropout))
#    model.add(Dense(32, activation='relu', kernel_initializer = 'he_normal'))
#    model.add(Dropout(dropout))
#    model.add(Dense(16, activation='relu', kernel_initializer = 'he_normal'))
#    model.add(Dropout(dropout))
#    model.add(Dense(16, activation='relu', kernel_initializer = 'he_normal'))
#    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))

    # opt = SGD(lr=0.01)
    opt = optimizers.Adam(learning_rate = 0.0001)                
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


# In[45]:


# Stop training if the loss (or any other metric) does not improve after a certain number of epochs in order to 
# avoid overfitting
# https://keras.io/api/callbacks/early_stopping/


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, mode="auto", verbose = 1)

# Create Keras Classifier and use predefined baseline model with early stopping
clf = KerasClassifier(build_fn = baseline_model, epochs = 1000, batch_size = 16, verbose = 1)
# Object to describe the training and validation result
# results = cross_val_score(clf, X_data, y_data, cv = cv, scoring = 'accuracy')
# results = cross_val_score(clf, X_data, y_data, cv = cv, scoring = 'accuracy', fit_params={'callbacks':early_stopping})
# results = cross_validate(clf, X_data, y_data, cv = cv)
results = clf.fit(X_data, y_data)
# Result
print (results)
print("Result: %.2f%% (%.2f# %%)" % (results.mean()*100, results.std()*100))


# In[46]:


clf.predict(X_data)


# In[47]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 500, class_weight = 'balanced', min_samples_split = 6, min_samples_leaf = 2)
results = cross_validate(clf, X_data, y_data, cv = cv)
print('Random Forest:', results['test_score'])


# In[48]:


# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

'''
clf = GradientBoostingClassifier(criterion = 'friedman_mse', learning_rate = 0.075,
                                 loss = 'deviance', max_depth = 14, max_leaf_nodes = None,
                                 min_samples_leaf = 5, min_samples_split = 8, n_estimators = 500,
                                 subsample = 0.75, min_weight_fraction_leaf = 0.1, min_impurity_decrease = 0)

'''
clf = GradientBoostingClassifier(criterion = 'friedman_mse', learning_rate = 0.26,
                                 loss = 'deviance', max_depth = 10, max_leaf_nodes = None,
                                 min_samples_leaf = 7, min_samples_split = 12, n_estimators = 100,
                                 subsample = 0.75, min_weight_fraction_leaf = 0.212, min_impurity_decrease = 0.1)

results = cross_validate(clf, X_data, y_data, cv = cv)
print('Gradient Boosting:', results['test_score'])


# In[49]:


# Multi Layer Perceptron (MLP)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', 
                    alpha=1e-15,
                    hidden_layer_sizes=(256,128,64),
                    activation="relu",
                    random_state=100)
results = cross_validate(clf, X_data, y_data, cv = cv)
print('Multi Layer Perceptron:', results['test_score'])


# In[50]:


# Cross validate prediction
# https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
y_pred = cross_val_predict(clf, X_data, y_data, cv = cv)
a_score = accuracy_score(y_data, y_pred)
r_score = r2_score(y_data, y_pred)
print ('Accuracy, R2 scores: ', a_score, r_score)


# In[51]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_data, y_pred, normalize = 'true')
cnf_matrix = np.array(cnf_matrix) * 100

#l = ['forest plantation', 'palm', 'natural forest' , 'agriculture', 'ground', 'aquaculture', 'sand', 'scattered forest', 'urban', 'water']
l = ['forest plantation', 'palm', 'natural forest' , 'non-forest']
hm = sns.heatmap(cnf_matrix, cmap = "GnBu", annot = True, fmt = '.1f', xticklabels = l, yticklabels = l)
hm.set(xlabel = '\npredicted', ylabel = 'true\n')
plt.xticks(rotation = 45)
plt.show()


# In[52]:


# https://medium.com/dataman-in-ai/a-wide-choice-for-modeling-multi-class-classifications-d97073ff4ec8
model = baseline_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1)
'''
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(random_state=1, max_iter=2000).fit(X_train,y_train)
y_pred = model.predict(X_test)
#score = model.evaluate(X_test, y_test,verbose=1)
#print(score)
#NN_pred = Prediction(NN_model)    
#pd.crosstab(y_pred,y_test).apply(lambda x: x/x.sum(), axis=1).round(2)
'''
from sklearn.metrics import r2_score
print(y_pred.shape)
print(y_test.shape)
accuracy_score(y_test, y_pred)
score = r2_score(y_test, y_pred)
print (score)


# In[53]:


y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test,verbose=1)
print(score)


# In[54]:


# Data Ingest
df = read_csv('y_pred_future_out.csv', header=None)


# In[55]:


df.values


# In[56]:


y_pred_future = clf.predict(df.values)


# In[57]:


y_pred_future


# In[58]:


np.savetxt("y_pred_future_landtype.csv", y_pred_future)

