#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright Sanchali Banerjee



# In[2]:


# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


# Supporting SHAP
# https://www.kaggle.com/code/phamvanvung/shap-for-lstm/notebook
# !pip install 'tensorflow==1.14.0'


# In[4]:


'''
The idea is to identify a time sequence of reflection values of a frequency band and determine the next reflection 
value for that frequency band in that sequence. It is assumed that the input dataset is grouped by location and 
ordered by date within each location.
For each location, the dataset provides reflection values for a sequence of 5 dates for each frequency band.
The first 4 dates provide a time sequence of reflection values of each frequency band and the 5th date provides
the expected reflection value for the corresponding frequency band. Each sequence is an entry in X and fifth is a
corresponding entry in y.
Since the only interest is in frequency reflection values for a group of 5 rows for a given frequency band, we can drop 
all columns that are not frequency values.
'''


# In[5]:


# Importing the training set
# Latitude (X), Longitude (Y), Date, LandType, X, Y, Band1,...,BandN
# Assume that the data is grouped by X, Y and sorted by Date within a group
dataset_train = pd.read_csv('~/Downloads/LandChange_Train.csv')
print (dataset_train)
# Remove columns names 'Date', 'Land_Type', 'X', 'Y'
# dataset_train = dataset_train.drop(['Date', 'Land_Type', 'X', 'Y'], axis=1)
# Select top 20 features
dataset_train_features = dataset_train[['band 171', 'band 76', 'band 87', 'band 40', 'band 37', 'band 116', 'band 58', 'band 10', 'band 11', 'band 100', 'band 48', 'band 86', 'band 73', 'band 28', 'band 39', 'band 232', 'band 4', 'band 130', 'band 54', 'band 223']]
dataset_train_labels = dataset_train[['Land_Type']] 
# Get the list of column names/features

features = list(dataset_train_features.columns)
print('Features')
print (features)
print('Feature Data')
train_features_set = dataset_train_features.values
print (train_features_set)
print('Label Data')
train_labels_set = dataset_train_labels.values
print (train_labels_set)


# In[6]:


X_train = train_features_set
X_train_labels = train_labels_set


# In[7]:


# Creating data structure for LSTM training (samples, timesteps, features) with 4 timesteps and 1 expected next timestep
feature_count = 20
timestep_count = 5
sample_count = X_train.shape[0] // timestep_count

print ('lstm dimensions:', sample_count, timestep_count, feature_count)
# Reshape to lstm dimensions
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
X_train_lstm = []
y_train_lstm = []
for i in range (0, sample_count,1):
    X_train_sample = X_train[i*timestep_count:(i+1)*timestep_count,:].reshape(1, timestep_count, feature_count)
    y_train_sample = X_train_labels[(i+1)*timestep_count - 1]
    if (i==0):
        X_train_lstm = X_train_sample
        print (X_train_sample.shape)
        print (X_train_sample)
        y_train_lstm = y_train_sample
        print (y_train_sample.shape)
        print (y_train_sample)
    else:
        # https://towardsdatascience.com/reshaping-numpy-arrays-in-python-a-step-by-step-pictorial-tutorial-aed5f471cf0b
        X_train_lstm = np.vstack((X_train_lstm, X_train_sample))
        y_train_lstm = np.vstack((y_train_lstm, y_train_sample))

print("X_train")
print (X_train_lstm.shape)
print (X_train_lstm)
print("y_train")
print (y_train_lstm.shape)
print (y_train_lstm)


# In[8]:


X_train = X_train_lstm
y_train = y_train_lstm


# In[9]:


print (X_train.shape)
print (y_train.shape)


# In[10]:


# Importing the test set
# Latitude (X), Longitude (Y), Date, LandType, X, Y, Band1,...,BandN
# Assume that the data is grouped by X, Y and sorted by Date within a group
dataset_test = pd.read_csv('~/Downloads/LandChange_Test.csv')
print (dataset_test)
# Remove columns names 'Date', 'Land_Type', 'X', 'Y'
# dataset_train = dataset_train.drop(['Date', 'Land_Type', 'X', 'Y'], axis=1)
# Select top 20 features

dataset_test_features = dataset_test[['band 171', 'band 76', 'band 87', 'band 40', 'band 37', 'band 116', 'band 58', 'band 10', 'band 11', 'band 100', 'band 48', 'band 86', 'band 73', 'band 28', 'band 39', 'band 232', 'band 4', 'band 130', 'band 54', 'band 223']]
dataset_test_labels = dataset_test[['Land_Type']] 

features = list(dataset_test_features.columns)
print('Features')
print (features)
print('Feature Data')
test_features_set = dataset_test_features.values
print (test_features_set)
print('Label Data')
test_labels_set = dataset_test_labels.values
print (test_labels_set)


# In[11]:


X_test = test_features_set
X_test_labels = test_labels_set


# In[12]:


# Creating data structure for LSTM test (samples, timesteps, features) with 4 timesteps and 1 expected next timestep
feature_count = 20
timestep_count = 5
sample_count = X_test.shape[0] // timestep_count

print ('lstm dimensions:', sample_count, timestep_count, feature_count)
# Reshape to lstm dimensions
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
X_test_lstm = []
y_test_lstm = []
for i in range (0, sample_count,1):
    X_test_sample = X_test[i*timestep_count:(i+1)*timestep_count,:].reshape(1, timestep_count, feature_count)
    y_test_sample = X_test_labels[(i+1)*timestep_count - 1]
    if (i==0):
        X_test_lstm = X_test_sample
        print (X_test_sample.shape)
        print (X_test_sample)
        y_test_lstm = y_test_sample
        print (y_test_sample.shape)
        print (y_test_sample)
    else:
        X_test_lstm = np.vstack((X_test_lstm, X_test_sample))
        y_test_lstm = np.vstack((y_test_lstm, y_test_sample))

print("X_test")
print (X_test_lstm.shape)
print (X_test_lstm)
print("y_test")
print (y_test_lstm.shape)
print (y_test_lstm)


# In[13]:


X_test = X_test_lstm
y_test = y_test_lstm


# In[14]:


print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[15]:


X_train


# In[16]:


X_test


# In[17]:


print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[18]:


# Importing the Keras Libraries and packages
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
'''
import tensorflow.compat.v1 as tf
# https://github.com/slundberg/shap/pull/1483
tf.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1.keras
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
'''
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
'''


# In[19]:


print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[41]:


# Initialising the RNN

input_timesteps=X_train.shape[1]
input_features=X_train.shape[2]
output_labels=y_train.shape[1]

regressor = Sequential()
# Adding the first LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = input_features, return_sequences = True, input_shape = (input_timesteps, input_features)))
regressor.add (Dropout(0.2))
# Adding a second LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = input_features,return_sequences = True))
regressor.add (Dropout(0.2))
# Adding a third LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = input_features, return_sequences = True))
regressor.add (Dropout(0.2))
# Adding a fourth LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = input_features, return_sequences = True))
regressor.add (Dropout(0.2))
# Adding the last LSTM Layer and some Dropout regularisation
regressor.add (LSTM(units = input_features, return_sequences = False))
regressor.add (Dropout (0.2))
# Changing the timesteps for the output layer
# regressor.add(RepeatVector(output_timesteps))
# Adding the outputLayer
#regressor.add(TimeDistributed(Dense(units = output_labels)))
regressor.add(Dense(units = output_labels))


# In[42]:


# Model information
regressor.summary()


# In[43]:


# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[44]:


# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 2000, batch_size = 8)


# In[45]:


print (X_test)
#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print (X_test.shape)
y_pred = regressor.predict(X_test)
print (y_pred.shape)
print(y_pred)
print (y_test.shape)
print(y_test)
# predicted_value = sc.inverse_transform(predicted_value)


# In[46]:

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
# score = r2_score(data["Actual Value"], data["Preds"])
score = r2_score(y_test, y_pred)
print("The r2_score of our model is {}%".format(round(score, 2) *100))


# In[47]:


# Visualising the results
plt.plot (y_test, color = 'red', label = 'Expected')
plt.plot (y_pred, color = 'blue', label = 'Predicted')
plt. title ('Frequency Band Value Prediction')
# plt.xlabel('Bands')
plt.ylabel ('Band Value')
plt.legend()
plt.show()


# In[48]:


import shap


# In[49]:



# In[50]:


# Use the training data for deep explainer => can use fewer instances
explainer = shap.DeepExplainer(regressor, X_train)
# explain the the testing instances (can use fewer instanaces)
# explaining each prediction requires 2 * background dataset size runs
shap_values = explainer.shap_values(X_test)
# init the JS visualization code
shap.initjs()


# In[51]:


explainer.expected_value


# In[52]:


len(shap_values)


# In[53]:


X_test.shape


# In[54]:


shap_values[0].shape


# In[55]:


shap_values[0][0].shape


# In[56]:


# shap.force_plot(explainer.expected_value[0], shap_values[0][0][0,:], features)
print(features)
print(len(features))


# In[57]:


i=0
j=0


# In[58]:


shap_values[0][i][j]


# In[59]:


X_test[i][j].shape


# In[60]:


# shap.force_plot(explainer.expected_value[0], shap_values[0][0], features)
i = 0
j = 0
x_test_df = pd.DataFrame(data=X_test[i][j].reshape(1,len(features)), columns = features)
shap.force_plot(explainer.expected_value[0], shap_values[0][i][j], x_test_df)


# In[61]:


################# Plot AVERAGE shap values for ALL observations  #####################
## Consider average (+ is different from -)
shap_average_value = shap_values[0].mean(axis=0)

x_average_value = pd.DataFrame(data=X_test.mean(axis=0), columns = features)
shap.force_plot(explainer.expected_value[0], shap_average_value, x_average_value)


# In[ ]:




