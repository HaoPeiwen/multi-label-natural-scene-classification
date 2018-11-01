from __future__ import print_function
from keras.layers.normalization import BatchNormalization
import csv
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
from skimage import io
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential

r = csv.reader(open('train.csv', 'r', encoding = 'utf8'))
x_test=[]
y_test=[]
x_train=[]
y_train = []
d = {}

for row in r:
    try:
        d[row[0]] = list(map(int, row[1:]))
    except:
        pass
print(len(d))
# split training set and testing set        
for file in os.listdir("train/"):
    x_train.append(io.imread("train/" + file))
    y_train.append(d[file.split('.')[0]])
for file in os.listdir("picture/"):
    x_test.append(io.imread("picture/" + file))
    y_test.append(d[file.split('.')[0]])
# ----


x_train = np.array(x_train).astype(np.float32)
y_train = np.array(y_train)
x_test = np.array(x_test).astype(np.float32)
y_test = np.array(y_test)
mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_test -= mean_image
x_train /= 128.
x_test /= 128.
#x_test /= 256.
#x_train /= 256.
nb_classes = 5


#base_model = keras.applications.resnet50.ResNet50(include_top=False, pooling='avg')
#base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, pooling='avg')
#base_model = keras.applications.MobileNetV2(include_top=False, pooling='avg')
#outputs = Dense(nb_classes, activation='sigmoid')(base_model.output)
#model = Model(base_model.inputs, outputs)

for path in ['MobileNetV2/22-0.24.hdf5', 'ResNet50/26-0.23.hdf5', 'InceptionResNetV2/30-0.22.hdf5']:
    model = keras.models.load_model(path)
    preds = (model.predict(x_test) >= 0.5).astype(int)
    all_test = np.sum(
        np.all(preds == y_test, axis=1)) / len(y_test)
    one_test = (np.sum(
        np.all(preds - y_test <= 0, axis=1)) / len(y_test)) - (np.sum(np.all(preds == 0, axis=1)) / len(y_test))

    print('\n'+path+': ')
    print('[Testing  set] binary :', np.sum(preds == y_test) / (5*len(y_test)))
    print('[Testing  set] all match:', all_test)
    print('[Testing  set] at least one match:', one_test,'\n')

