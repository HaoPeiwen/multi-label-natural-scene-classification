from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

import numpy as np
import os
import csv
from skimage import io
import pickle

r = csv.reader(open('train.csv', 'r', encoding='utf8'))
x_test=[]
y_test=[]
x_train=[]
y_train=[]
d = {}
for row in r:
    try:
        d[row[0]] = list(map(int, row[1:]))
    except:
        pass
for file in os.listdir("train/"):
    x_train.append(io.imread("train/" + file).flatten())
    y_train.append(d[file.split('.')[0]])
for file in os.listdir("test/"):
    x_test.append(io.imread("test/" + file).flatten())
    y_test.append(d[file.split('.')[0]])

x_train=np.asarray(x_train).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)
y_test=np.asarray(y_test)
y_train=np.asarray(y_train)
mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_test -= mean_image
x_train /= 128.
x_test /= 128.

classifier = OneVsRestClassifier(SVC()) 
classifier.fit(x_train, y_train)

with open('onevsrest-svc-rbf-rgb.pkl', 'wb') as f:
    pickle.dump(classifier, f)

'''
with open('onevsrest-svc-rbf-rgb.pkl', 'rb') as f:
    classifier = pickle.load(f)
'''
predictions = classifier.predict(x_test)
print('all match:', np.sum(np.all(predictions ==y_test, axis=1)) / len(y_test))
print('at least one match:', (np.sum(np.all(predictions - y_test <= 0, axis=1))-np.sum(np.all(predictions== 0, axis=1))) / len(y_test))
print('binary :', np.sum(predictions==y_test) / (5*len(y_test)))
