!pip install python-gdcm --no-index --find-links=file:///kaggle/input/pythongdcm/
!pip install pylibjpeg --no-index --find-links=file:///kaggle/input/pylibjpeg/

# Import the required packages
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import cv2
import pydicom
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
from io import BytesIO

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator


# Input Folders
input_path = "/kaggle/input/rsna-breast-cancer-detection/"

# Output Folders
output_path = '/kaggle/working/rsna-breast-cancer-detection/'
model_path = '/kaggle/input/cnn-contest-model/'

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)


# Define target image parameters
img_size = 128
num_channels = 1


# Define a dicom image conversion

def convert_dicom_pixels (filename):
    # print("Inside Convert Dicom")
    # Get the DICOM image
    dcm_img = pydicom.dcmread (filename)
    # Convert DICOM into multi-dimensional array of pixel intensity values
    img = dcm_img.pixel_array
    # Resize

    img = resize(img,(img_size,img_size))
    # Scale the pixel intensity values
    img = img.astype(float)
    img = (np.maximum(img, 0) / img.max()) * 255.0
    img = np.uint8(img)
    
    return img


# Input Folders for test
test_images_path = "/kaggle/input/rsna-breast-cancer-detection/test_images/"


# Get test data from the source
test_df = pd.read_csv(input_path + 'test.csv')
# test_df = pd.read_csv(input_path + 'test.csv', nrows=16)
test_df['image_rel_path'] = test_df['patient_id'].astype(str) +'/'+test_df['image_id'].astype(str) + '.dcm'
test_df['image_new'] = test_df['patient_id'].astype(str) +'_'+test_df['image_id'].astype(str) + '.png'
###print(test_df.shape)
###test_df.head()


# Create output folders for test
output_test_images_path = '/kaggle/working/rsna-breast-cancer-detection/test_images/'
# https://stackoverflow.com/questions/70235696/checking-folder-and-if-it-doesnt-exist-create-it
if not os.path.exists(output_test_images_path):
    os.makedirs(output_test_images_path)


# load model
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
from tensorflow.keras.models import load_model
model = load_model(model_path + 'model.h5')
###print("Loaded model from disk")




import random

# Iterate over each test image and predict
# https://stackoverflow.com/questions/43017017/keras-model-predict-for-a-single-image
test_df = test_df.reset_index(drop = True)

predict = np.empty((0,2), dtype = float)

for row_idx, row in test_df.iterrows():
###for row_idx, row in tqdm(test_df.iterrows()):
 
    # Predict image
    # Get the DICOM image
    dcm_file = test_images_path + test_df['image_rel_path'][row_idx]
    # Convert the DICOM image to pixels
    img = convert_dicom_pixels (dcm_file)
   
    # Understand an example of the converted image from some row
    ###if row_idx == 2:
        ###plt.imshow(img, cmap = 'gray_r')

    # Make it the correct tensor rank
    img = img.reshape(img_size, img_size, num_channels)
    img = np.expand_dims(img, axis = 0)
    
    predict1 = model.predict(img)
        
    # Append to the prediction list
    # https://www.geeksforgeeks.org/how-to-append-two-numpy-arrays/
    predict = np.append(predict, predict1, axis = 0)
    
### predict[:5]


# Predict the class label
# y_test_classes = predict.argmax(axis=-1)
y_test_classes = np.round(predict[:,1],2)
###print(y_test_classes)

# Output submission
# https://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-and-then-append-to-it-in-numpy
pred_df = test_df['prediction_id']

y_test_classes_df = pd.DataFrame(y_test_classes[:],columns = ['prediction'])
###print(y_test_classes_df)

pred_df = pd.concat([pred_df, y_test_classes_df], axis=1)
###print (pred_df)
# Using reset_index to properly position the column headings
# Output maximum probability of cancer for each prediction id
pred_df = pred_df.groupby('prediction_id').max('prediction').reset_index()
# Rename prediction ot cancer for submission
pred_df.rename(columns = {'prediction':'cancer'}, inplace = True)
###print (pred_df)
pred_df.to_csv(path_or_buf='submission.csv', index=False)
