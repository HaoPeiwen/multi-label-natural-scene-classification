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
# split training set and testing set        
for file in os.listdir("train/"):
    x_train.append(io.imread("train/" + file))
    y_train.append(d[file.split('.')[0]])
for file in os.listdir("test/"):
    x_test.append(io.imread("test/" + file))
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


modelname = 'Sequential_BN'
try:
    os.mkdir(modelname)
except:
    pass
#base_model = keras.applications.resnet50.ResNet50(include_top=False, pooling='avg')
#base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, pooling='avg')
#base_model = keras.applications.MobileNetV2(include_top=False, pooling='avg')
#outputs = Dense(nb_classes, activation='sigmoid')(base_model.output)
#model = Model(base_model.inputs, outputs)



lr_reducer = ReduceLROnPlateau(factor=np.sqrt(
    0.1), cooldown=0, patience=5, min_lr=0.5e-6)
#early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger(modelname + '/BinaryAccuracy.csv')


# instantiate model
model = Sequential()


model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
#model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))






model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images


# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
class Accuracy(Callback):
    def __init__(self, val_x, val_y, x, y):
        self.val_x = val_x
        self.val_y = val_y
        self.x = x
        self.y = y
        self.all_test = 0.0
        self.all_train = 0.0
        self.one_test = 0.0
        self.one_train = 0.0
        self.header = True

    def on_epoch_end(self, epoch, logs={}):
        preds = (self.model.predict(self.val_x) >= 0.5).astype(int)
        preds_train = (self.model.predict(self.x) >= 0.5).astype(int)
        self.all_test = np.sum(
            np.all(preds == self.val_y, axis=1)) / len(self.val_y)
        self.all_train = np.sum(
            np.all(preds_train == self.y, axis=1)) / len(self.y)
        self.one_test = (np.sum(
            np.all(preds - self.val_y <= 0, axis=1)) / len(self.val_y)) - (np.sum(np.all(preds == 0, axis=1)) / len(self.val_y))
        self.one_train = (np.sum(
            np.all(preds_train - self.y <= 0, axis=1)) / len(self.y)) - (np.sum(np.all(preds_train == 0, axis=1)) / len(self.y))
        print('\n[Testing  set] all match:', self.all_test)
        print('[Training set] all match:', self.all_train)
        print('[Testing  set] at least one match:', self.one_test)
        print('[Training set] at least one match:', self.one_train, '\n')
        

        with open(modelname + '/MatchAccuracy.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if self.header:
                writer.writerow(['epoch', 'valAll', 'trainAll',
                                 'valAtLeastOne', 'trainAtLeastOne'])
                self.header = False
            writer.writerow([epoch,self.all_test, self.all_train, self.one_test, self.one_train])



        


filepath = modelname+"/{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True,
                                             save_weights_only=False, mode='max', period=1)
datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    validation_data=(x_test, y_test),
                    epochs=100, verbose=1,
                    callbacks=[lr_reducer, csv_logger, checkpoint, Accuracy(x_test, y_test, x_train, y_train)])


"""
model = load_model('20-0.30.hdf5')
for i in [493, 501, 502, 503, 893, 894, 895]:
    p = io.imread("img/%d.jpg" % i)
    p = np.asarray(p).astype(np.float32)
    p -= mean_image
    p /= 128.
    p = np.expand_dims(p, axis=0)
    print(i, model.predict(p))
"""
