from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import csv
from skimage import io
import cv2
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
    y_train.append(d[file.split('.')[0]])
    x_luv=cv2.cvtColor(cv2.imread("train/" + file),cv2.COLOR_RGB2LUV)
    x_luv=x_luv-128
    x_luv=x_luv/256
    l=np.asarray([])
    for i in range(7):
        for j in range(7):
            block= x_luv[32*i:32*(i+1),32*j:32*(j+1)]
            mean,var= np.mean(block, axis=tuple(range(block.ndim-1))),np.var(block, axis=tuple(range(block.ndim-1)))
            l=np.concatenate((l,mean))
            l=np.concatenate((l,var))
    x_train.append(l)
for file in os.listdir("test/"):
    y_test.append(d[file.split('.')[0]])
    x_luv=cv2.cvtColor(cv2.imread("test/" + file),cv2.COLOR_RGB2LUV)
    x_luv=x_luv-128
    x_luv=x_luv/256

    l=np.asarray([])
    for i in range(7):
        for j in range(7):
            block= x_luv[32*i:32*(i+1),32*j:32*(j+1)]
            mean,var= np.mean(block, axis=tuple(range(block.ndim-1))),np.var(block, axis=tuple(range(block.ndim-1)))
            l=np.concatenate((l,mean))
            l=np.concatenate((l,var))
    x_test.append(l)
x_train=np.asarray(x_train).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)
y_test=np.asarray(y_test)
y_train=np.asarray(y_train)
print (x_train.shape,y_train.shape,x_test.shape,y_test.shape)

classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
classifier.fit(x_train, y_train)

with open('onevsrest-knn-3-luv.pkl', 'wb') as f:
    pickle.dump(classifier, f)

'''
with open('onevsrest-knn-3-luv.pkl', 'rb') as f:
    classifier = pickle.load(f)
'''
predictions = classifier.predict(x_test)
print('all match:', np.sum(np.all(predictions ==y_test, axis=1)) / len(y_test))
print('at least one match:', (np.sum(np.all(predictions - y_test <= 0, axis=1))-np.sum(np.all(predictions== 0, axis=1))) / len(y_test))
print('binary :', np.sum(predictions==y_test) / (5*len(y_test)))
