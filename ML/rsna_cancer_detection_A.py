# %% [code] {"jupyter":{"outputs_hidden":false}}
!pip install python-gdcm --no-index --find-links=file:///kaggle/input/pythongdcm/
!pip install pylibjpeg --no-index --find-links=file:///kaggle/input/pylibjpeg/

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Enable TPU
# https://www.kaggle.com/docs/tpu
tpu_enabled = True

# %% [code] {"jupyter":{"outputs_hidden":false}}
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Input Folders
input_path = "/kaggle/input/rsna-breast-cancer-detection/"

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Output Folders
output_path = '/kaggle/working/rsna-breast-cancer-detection/'
model_path = '/kaggle/working/rsna-breast-cancer-detection/model/'

# https://stackoverflow.com/questions/70235696/checking-folder-and-if-it-doesnt-exist-create-it
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Define target image parameters
img_size = 128
# CHNL num_channels = 3
num_channels = 1

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Define a dicom image conversion

def convert_dicom_pixels (filename):
    # print("Inside Convert Dicom")
    # Get the DICOM image
    dcm_img = pydicom.dcmread (filename)
    # Convert DICOM into multi-dimensional array of pixel intensity values
    # CHNL img = dcm_img.pixel_array.astype(float)
    img = dcm_img.pixel_array
    # Resize
    # CHNL img = resize(img,(img_size,img_size,num_channels))
    img = resize(img,(img_size,img_size))
    # Scale the pixel intensity values
    # CHNL 
    img = img.astype(float)
    img = (np.maximum(img, 0) / img.max()) * 255.0
    img = np.uint8(img)
    
    return img

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Define a dicom image conversion

def convert_dicom (filename):
    img = convert_dicom_pixels (filename)

    img = Image.fromarray(img)
    
    # Write PIL Image to in-memory PNG

    membuf = BytesIO()
    img.save(membuf, format="png") 
    img = Image.open(membuf)

    return img

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Input training folders
train_images_path = "/kaggle/input/rsna-breast-cancer-detection/train_images/"

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Get training data from the source and populate the dataframes
train_df = pd.read_csv(input_path + 'train.csv')
# train_df['image_fullpath'] = train_images_path + train_df['patient_id'].astype(str) +'/'+train_df['image_id'].astype(str) + '.dcm'
train_df['image_rel_path'] = train_df['patient_id'].astype(str) +'/'+train_df['image_id'].astype(str) + '.dcm'
train_df['image_new'] = train_df['patient_id'].astype(str) +'_'+train_df['image_id'].astype(str) + '.png'

# Check if this should be included 
train_df['cancer'] = train_df['cancer'].astype(str)

print(train_df.shape)
train_df.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Check some training images
dcm_img = pydicom.dcmread(train_images_path + train_df['image_rel_path'][2])
dcm_img
img = dcm_img.pixel_array
plt.imshow(img, cmap = 'gray_r')

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Create a reasonable mix of all categories so that all categories are learnt
# The reasonable mix is created since processing on 50000+ images takes a long time

# Category 1: Non-cancer
category1_count_param = 2000
# Category 2: Cancer
category2_count_param = 1000

# filter for non-cancer
filter1_df = train_df[(train_df['cancer'] == '0')]
print (filter1_df.shape)
# filter for cancer
filter2_df = train_df[(train_df['cancer'] == '1')]
print (filter2_df.shape)

# samples from filtered data
sample1_df = filter1_df.sample(category1_count_param)
sample2_df = filter2_df.sample(category2_count_param)

# combine the sampled rows
df = sample1_df.append(sample2_df)
# Randomly reshuffle the sampled rows
df = df.sample(frac = 1)

# check if the total number of samples shown
# sample1_df.head()
# sample2_df.head()
print ('sample 1, sample 2, total sample:', sample1_df.shape, sample2_df.shape, df.shape)
df.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Make the sample the new training dataset
train_df = df
print (train_df.shape)
train_df.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
# Create training output folders
output_train_images_path = '/kaggle/working/rsna-breast-cancer-detection/train_images/'


if not os.path.exists(output_train_images_path):
    os.makedirs(output_train_images_path)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Convert DICOM images to png in output folder


train_df = train_df.reset_index(drop = True)

for row_idx, row in tqdm(train_df.iterrows()):
    # print(row['patient_id'], row['image_id'], row['path'])
    # print('Row Id', row_idx)
    
    
    # Get the DICOM image
    dcm_file = train_images_path + train_df['image_rel_path'][row_idx]
    # Convert the DICOM image
    img = convert_dicom (dcm_file)
    
    # Save the converted image
    image_file = output_train_images_path + train_df['image_new'][row_idx]
    # cv2.imwrite(image_file, img)
    img.save(image_file)
    
    # Understand an example of the converted image from some row
    # https://www.geeksforgeeks.org/python-working-with-png-images-using-matplotlib/
    if row_idx == 2: 
        plt.imshow(img, cmap = 'gray_r')

# %% [code] {"jupyter":{"outputs_hidden":false}}
# TPU Strategy
if (tpu_enabled == True):
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Initialize Keras’ ImageDataGenerator class
validation_split = 0.2

train_datagen = ImageDataGenerator(
        # rescale=1 / 255.0,
        # rotation_range=20,
        # zoom_range=0.05,
        # width_shift_range=0.05,
        # height_shift_range=0.05,
        # shear_range=0.05,
        # horizontal_flip=True,
        # fill_mode="nearest",
        validation_split = validation_split)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Initialize our training, validation generator
if (tpu_enabled):
    batch_size = 16 * tpu_strategy.num_replicas_in_sync
else:
    batch_size = 8
    
img_code = 'image_new'
# img_code = 'image_fullpath'
target = 'cancer'

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=output_train_images_path,
    x_col=img_code,
    y_col=target,
    # CHNL 
    color_mode="grayscale",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    # class_mode="raw",
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42,
    validate_filenames=False,
)
validation_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=output_train_images_path,
    x_col=img_code,
    y_col=target,
    # CHNL 
    color_mode="grayscale",
    target_size=(img_size, img_size),
    batch_size=1,
    # batch_size=batch_size,
    # class_mode="raw",
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=42,
    validate_filenames=True,
)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Define the Convolutional Neural Network (CNN)

def prepare_model():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(img_size, img_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Train the model using fit_generator

if (tpu_enabled):
    # instantiating the model in the strategy scope creates the model on the TPU
    with tpu_strategy.scope():
        model = prepare_model()
else:
    model = prepare_model()


history = model.fit(
                    train_generator,
                    steps_per_epoch=train_generator.n//train_generator.batch_size,
                    epochs=5,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n//validation_generator.batch_size,
                    # class_weight=class_weights,
                    # initial_epoch=init_epoch_train,
                    max_queue_size=15,
                    workers=8,
                    use_multiprocessing=True
                    # callbacks=callbacks_list
                    )

# %% [code] {"jupyter":{"outputs_hidden":false}}
'''
# Evaluate our model performance

score = model.evaluate(validation_generator)
print('Validation test loss:', score[0])
print('Validation test accuracy:', score[1])
'''

# %% [code] {"jupyter":{"outputs_hidden":false}}

'''
# serialize model to JSON
model_json = model.to_json()
with open(model_path + "model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_path + "model.h5")
'''
#model.save(model_path + "model.h5")
from tensorflow.keras.models import save_model
save_model(model, model_path + "model.h5")
print("Saved model to disk")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Input Folders for test
test_images_path = "/kaggle/input/rsna-breast-cancer-detection/test_images/"

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Get test data from the source

test_df = pd.read_csv(input_path + 'test.csv')
# test_df = pd.read_csv(input_path + 'test.csv', nrows=16)
test_df['image_rel_path'] = test_df['patient_id'].astype(str) +'/'+test_df['image_id'].astype(str) + '.dcm'
test_df['image_new'] = test_df['patient_id'].astype(str) +'_'+test_df['image_id'].astype(str) + '.png'
###print(test_df.shape)
###test_df.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Create output folders for test
output_test_images_path = '/kaggle/working/rsna-breast-cancer-detection/test_images/'

if not os.path.exists(output_test_images_path):
    os.makedirs(output_test_images_path)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Convert DICOM images to png in output folder


test_df = test_df.reset_index(drop = True)

for row_idx, row in tqdm(test_df.iterrows()):
    # print(row['patient_id'], row['image_id'], row['path'])
    # print('Row Id', row_idx)
    
    # Get the DICOM image
    dcm_file = test_images_path + test_df['image_rel_path'][row_idx]
    # Convert the DICOM image
    img = convert_dicom (dcm_file)
    
    # Save the converted image
    image_file = output_test_images_path + test_df['image_new'][row_idx]
    # cv2.imwrite(image_file, img)
    img.save(image_file)
    

    if row_idx == 2: 
        plt.imshow(img, cmap = 'gray_r')

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Initialize Keras’ ImageDataGenerator class for test
img_code = 'image_new'
test_datagen = ImageDataGenerator()
# test_datagen = ImageDataGenerator(rescale=1 / 255.0)
# test_datagen = ImageDataGenerator(rescale=1 / 255.0, preprocessing_function = convert_dicom)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    # directory=test_images_path,
    directory=output_test_images_path,
    x_col=img_code,
    y_col=None,
    # CHNL 
    color_mode="grayscale",
    target_size=(img_size, img_size),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    validate_filenames=False,
)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# load model
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
'''
from tensorflow.keras.models import model_from_json
json_file = open(model_path + 'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_path + 'model.h5')
###print("Loaded model from disk")
# compile model
loaded_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
###print("Compiled model")
'''
from tensorflow.keras.models import load_model
model = load_model(model_path + 'model.h5')
print("Loaded model from disk")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Make a prediction on test data using Keras’ predict_generator
test_generator.reset()
predict=model.predict(test_generator, steps = len(test_generator.filenames))

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Keras’ predict_generator return the class probability of each class
predict[:5]

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Iterate over each test image and predict

test_df = test_df.reset_index(drop = True)

predict = np.empty((0,2), dtype = float)
for row_idx, row in tqdm(test_df.iterrows()):
    
    # Get the DICOM image
    dcm_file = test_images_path + test_df['image_rel_path'][row_idx]
    # Convert the DICOM image to pixels
    img = convert_dicom_pixels (dcm_file)
   
    # Understand an example of the converted image from some row

    if row_idx == 2: 
        plt.imshow(img, cmap = 'gray_r')

    # Make it the correct tensor rank
    # CHNL 
    img = img.reshape(img_size, img_size, num_channels)
    img = np.expand_dims(img, axis = 0)
    
    # Predict image
    predict1 = model.predict(img)
    # Append to the prediction list

    predict = np.append(predict, predict1, axis = 0)
    
predict[:5]

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Predict the class label
# y_test_classes = predict.argmax(axis=-1)
y_test_classes = np.round(predict[:,1],2) 
print(y_test_classes)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Output submission

pred_df = test_df['prediction_id']
# https://stackoverflow.com/questions/8386675/extracting-specific-columns-in-numpy-array
y_test_classes_df = pd.DataFrame(y_test_classes[:],columns = ['prediction'])
print(y_test_classes_df)

pred_df = pd.concat([pred_df, y_test_classes_df], axis=1)
print (pred_df)

# Using reset_index to properly position the column headings
# Output maximum probability of cancer for each prediction id
pred_df = pred_df.groupby('prediction_id').max('prediction').reset_index()
# Rename prediction ot cancer for submission

pred_df.rename(columns = {'prediction':'cancer'}, inplace = True)
print (pred_df)
pred_df.to_csv(path_or_buf='submission.csv', index=False)