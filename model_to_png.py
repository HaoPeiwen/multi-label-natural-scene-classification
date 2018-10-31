from keras.utils import plot_model
from keras import models
import os

#for modelname in ['Sequential_noBN', 'Sequential_BN', 'ResNet50', 'MobileNetV2', 'InceptionResNetV2']:
for modelname in ['Sequential_noBN', 'Sequential_BN']:
    f_list = os.listdir(modelname+'/')
    for i in f_list:
        if os.path.splitext(i)[1] == '.hdf5':
            model = models.load_model(modelname + '/' + i)
            print(model.summary())
            plot_model(model, to_file=modelname + '.png', show_shapes=True)
