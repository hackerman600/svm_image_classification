import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score


#load in the data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train[:20000]
y_train = y_train[:20000]

#check if the data is the one we wanted 
imshow(x_train[0],cmap="gray")
plt.show()

#normalise the data 
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print("x_train.shape = ",x_train.shape)

#use svm classifier.
svm_classifier = svm.SVC()

#fit classifier
svm_classifier.fit(x_train,y_train)

#make predictions
y_pred = svm_classifier.predict(x_test)

#check accuracy
accuracy = accuracy_score(y_test,y_pred)
print("accuracy is: ", accuracy*100, "%")
