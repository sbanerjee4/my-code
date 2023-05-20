#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Copyright Sanchali Banerjee


# In[3]:


# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
dataset_train = dataset_train.drop(['Date', 'Land_Type', 'X', 'Y'], axis=1)
#training_set = dataset_train.iloc[:, 1:2].values
training_set = dataset_train.values
print (training_set)


# In[6]:



# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler (feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# In[7]:


timesteps_count = 4
band_count = 235
training_set_count = len (training_set)
X_train = []
y_train = []

# Take a collection of timesteps_count + 1 rows and then for each band create a horizontal entry of timesteps 
# in X_train and an expected value in y_train.
for i in range (timesteps_count, training_set_count, 5):
    for j in range (0, band_count):
        X_train.append (training_set_scaled[i-4:i, j])
        y_train.append (training_set_scaled[i, j])
        
X_train, y_train = np.array(X_train), np.array(y_train)
print (X_train[0], y_train[0])
print (X_train[2], y_train[2])

# Reshaping
print (X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print (X_train.shape)
print (y_train.shape)


# In[8]:


# Importing the test set
# Latitude, Longitude, Date, LandType, X, Y, Band1,...,BandN
# Assume that the data is sorted by Latitude, Longitude, Date
dataset_test = pd.read_csv('~/Downloads/LandChange_Test.csv')
print (dataset_test)

# Remove columns 'Date', 'X', 'Y'. Do not drop Land_Type since it will be the expected value of the land change
# dataset_test = dataset_test.drop(['Date', 'Land_Type', 'X', 'Y'], axis=1)
dataset_test = dataset_test.drop(['Date', 'X', 'Y'], axis=1)
#training_set = dataset_train.iloc[:, 1:2].values
test_set = dataset_test.values
print (test_set)


# In[9]:



# Feature scaling for test
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler (feature_range = (0, 1))
test_set_scaled = sc.fit_transform(test_set)


# In[10]:


# Creating a data structure with 4 timesteps and 1 output for expected test
timesteps_count = 4
output_count = 1
band_count = 235
test_set_count = len (test_set)
print ('test_set_count: ', test_set_count)
X_test = []
y_test = []

# Collect test data for the last test date for predicting land types
# Each row in X has all the bands, each row in Y has a land type
X_test_last_landtype = []
y_test_last_landtype = []

# Collect test data for the first test date of the time series
X_test_first_landtype = []
y_test_first_landtype = []

# Take a collection of timesteps_count + 1 rows and then for each band create a horizontal entry of timesteps_count 
# in X_test and an expected value in y_test.

for i in range (timesteps_count, test_set_count, 5):
    
    # Append Bands and Landtype at the end of the time interval
    X_test_last_landtype.append (test_set_scaled[i,1:band_count+1])
    y_test_last_landtype.append (test_set_scaled[i,0])
    # Append Bands and Landtype at the start of the time interval
    X_test_first_landtype.append (test_set_scaled[i-timesteps_count,1:band_count+1])
    y_test_first_landtype.append (test_set_scaled[i-timesteps_count,0])
    
    for j in range (0, band_count):
        # If (Date, Land_Type, X, Y) are not dropped, then bands start at column 4 (j+4) else 0 (j)
        X_test.append (test_set_scaled[i-4:i, j+1])
        y_test.append (test_set_scaled[i, j+1])

        
X_test, y_test = np.array(X_test), np.array(y_test)
print (X_test[0], y_test[0])
print (X_test[10], y_test[10])

X_test_first_landtype, y_test_first_landtype = np.array(X_test_first_landtype), np.array(y_test_first_landtype)
print (X_test_first_landtype[0], y_test_first_landtype[0])
print (X_test_first_landtype[5], y_test_first_landtype[5])

X_test_last_landtype, y_test_last_landtype = np.array(X_test_last_landtype), np.array(y_test_last_landtype)
print (X_test_last_landtype[0], y_test_last_landtype[0])
print (X_test_last_landtype[5], y_test_last_landtype[5])

# Reshaping test
print (X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print (X_test.shape)
print (y_test.shape)
print (y_test)

# Landtype test
print (X_test_first_landtype.shape)
print (y_test_first_landtype.shape)

print (X_test_first_landtype)
print (y_test_first_landtype)

print (X_test_last_landtype.shape)
print (y_test_last_landtype.shape)

print (X_test_last_landtype)
print (y_test_last_landtype)


# In[11]:


# Importing the Keras Libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[12]:


# Initialising the RNN
regressor = Sequential()


# In[13]:


# Adding the first LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add (Dropout(0.2))
# Adding a second LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add (Dropout(0.2))
# Adding a third LSTM Layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add (Dropout(0.2))
# Adding a fourth LSTM Layer and some Dropout regularisation
regressor.add (LSTM(units = 50))
regressor.add (Dropout (0.2))


# In[14]:


# AddingtheoutputLayer
regressor.add(Dense(units = 1))


# In[15]:


# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[16]:


# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 1000, batch_size = 32)


# In[17]:


'''
# Getting the predicted stock price of 2017
print(dataset_train.shape, dataset_test.shape)
dataset_total = pd.concat ((dataset_train['Open'], dataset_test['Open']), axis = 0)
print (dataset_total.shape)
inputs = dataset_total[len (dataset_total) - len (dataset_test) - 60: ].values
print (inputs.shape)
inputs = inputs.reshape(-1,1)
print (inputs.shape)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append (inputs[i-60:i, 0])
'''
print (X_test)
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print (X_test.shape)
y_pred = regressor.predict(X_test)
y_pred = y_pred.reshape(-1)
print(y_pred)
print(y_test)
# predicted_value = sc.inverse_transform(predicted_value)


# In[18]:


# https://www.freecodecamp.org/news/evaluation-metrics-for-regression-problems-machine-learning/
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
# score = r2_score(data["Actual Value"], data["Preds"])
score = r2_score(y_test, y_pred)
print("The r2_score of our model is {}%".format(round(score, 2) *100))


# In[19]:


# Visualising the results
plt.plot (y_test, color = 'red', label = 'Expected')
plt.plot (y_pred, color = 'blue', label = 'Predicted')
plt. title ('Frequency Band Value Prediction')
# plt.xlabel('Bands')
plt.ylabel ('Band Result')
plt.legend()
plt.show()


# In[83]:


# Show selected bands
# https://www.adamsmith.haus/python/answers/how-to-subsample-every-nth-entry-in-a-numpy-array-in-python
print ("Band Count: ", band_count)
# Create x axis with bands
# https://favtutor.com/blogs/how-to-initialize-an-array-in-python
x_select = []
x_select = [i for i in range(band_count)] 
print(x_select)
    
for i in range (1, 24, 1):
    location_select = i
    y_test_select = y_test[(location_select - 1)*band_count:location_select * band_count]
    y_pred_select = y_pred[(location_select - 1)*band_count:location_select * band_count]
    # Visualising the results
    plt.plot (y_test_select, color = 'red', label = 'Actual')
    plt.plot (y_pred_select, color = 'blue', label = 'Predicted')
    # plt.scatter(x_select, y_test_select, c ="red", s = 2, label = 'Expected')
    # plt.scatter(x_select, y_pred_select, c ="blue", s = 2, label = 'Predicted')
    plt. title ('Location ' + str(location_select))
    plt.xlabel('Band')
    plt.ylabel ('Reflectance')
    plt.legend()
    # https://stackabuse.com/how-to-set-axis-range-xlim-ylim-in-matplotlib/
    plt.ylim([-1000, 4500])
    plt.show()


# In[84]:


X_test_first_landtype


# In[85]:


y_test_first_landtype


# In[86]:


X_test_last_landtype


# In[87]:


y_test_last_landtype


# In[88]:


y_pred.shape


# In[89]:


X_test_last_landtype.shape


# In[90]:


y_pred_out = np.reshape(y_pred, (X_test_last_landtype.shape[0], X_test_last_landtype.shape[1]))


# In[91]:


y_pred_out


# In[92]:


y_pred


# In[93]:


y_pred_out


# In[94]:


np.savetxt("y_pred_out.csv", y_pred_out, delimiter=",")


# In[95]:


np.savetxt("y_test_first_landtype.csv", y_test_first_landtype)


# In[96]:


y_test_first_landtype

