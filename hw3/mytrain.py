#from __future__ import print_function  # mode = 3
import numpy as np
#import matplotlib.pyplot as plt
import sys
import csv
from math import floor
from random import shuffle
import os  #mode = 3

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback

# Working on different workstation
#mode = 3

'''if mode == 1:
	path = os.environ.get("GRAPE_DATASET_DIR")
	print(path)
	traincsv = os.path.join(path,"train.csv")
	testcsv = os.path.join(path,"test.csv")
	outputfile = "output"
elif mode == 2:
	path = os.environ.get("HOME")
	path = path + '/Desktop/ML/hw3'
	traincsv = os.path.join(path,"train.csv")
	testcsv = os.path.join(path,"test.csv")
	outputfile = os.path.join(path,"output_test")
else:
	traincsv = sys.argv[1]
	#testcsv = sys.argv[2]
	outputfile = sys.argv[2]'''
traincsv = sys.argv[1]
outputfile = sys.argv[2]

#print(traincsv)
#print(testcsv)
#print(outputfile)

print(">>>>> Loading Data ...")

# loading training data

traincsv = open(traincsv,'r')
temp = csv.reader(traincsv,delimiter = ',')
label = []
train = []
rownum = 0
for r in temp:
	if rownum!=0:
		l = [0,0,0,0,0,0,0]
		l[int(r[0])]=1
		label.append(l)
		train.append(r[1].split())
	rownum+=1
traincsv.close()
train = np.array(train)
train = train.astype(float)
label = np.array(label)

# loading testing data

print("Size of label    set: "+str(label.shape))
print("Size of training set: "+str(train.shape))

print("\n>>>>> Normalize the Data ...")
train /= 255

print("\n>>>>> Shuffle and Split the Validation Set ...")
def _shuffle(X,Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize],Y[randomize])

def split_valid_set(X_all, Y_all, percentage):
	all_size = len(X_all)
	N_train =  int(floor(all_size*percentage))
	X_all, Y_all = _shuffle(X_all, Y_all)
	X_train, Y_train = X_all[0:N_train], Y_all[0:N_train]
	X_valid, Y_valid = X_all[N_train:] , Y_all[N_train:]
	return X_train, Y_train, X_valid, Y_valid

percentage = 0.6 # training set portion
X_train,Y_train,X_valid,Y_valid = split_valid_set(train,label,percentage)
X_train = X_train.reshape(len(X_train),48,48,1)
X_valid = X_valid.reshape(len(X_valid),48,48,1)

print("Size of training set: "+str(X_train.shape))
print("Size of validation  : "+str(X_valid.shape))

print("\n>>>>> Generate shift and rotated Image Data ...")
datagen = ImageDataGenerator(rotation_range = 20,zoom_range=0.2,width_shift_range=0.2,
							height_shift_range=0.2,horizontal_flip=True)
datagen.fit(X_train)
t = 0
for X_batch,Y_batch in datagen.flow(X_train,Y_train,batch_size=1000):
	if t < 200:
		X_train = np.concatenate((X_train,X_batch), axis = 0)
		Y_train = np.concatenate((Y_train,Y_batch), axis = 0)
	else: break
	t+=1
print("Size of training data : "+str(X_train.shape))
print("Size of training label: "+str(Y_train.shape))

#print("0:angry 1:hate 2:fear 3:happy 4:sad 5:surprise 6:X")

print("\n>>>>> Build Covolution ...")
model = Sequential()
model.add(Conv2D(64, (3,3), padding = 'valid', activation = 'relu', input_shape = (48,48,1)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))



model.add(Flatten())
#model.add(Dense(units = 128,activation = 'relu'))
#model.add(Dense(units = 64,activation = 'relu'))
model.add(Dense(units = 64,activation = 'relu'))
model.add(Dense(units = 7, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

score = model.evaluate(X_valid,Y_valid)
print("\nAccuracy of Validation set is: "+str(score[1]))

if not os.path.exists(outputfile):
	os.makedirs(outputfile)
filepath = outputfile + "/model-{val_acc:.5f}.h5"

checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose = 1,save_best_only=True,mode='max') 
batch_print_callback = LambdaCallback(on_epoch_end=lambda batch, logs:
						print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %
						(batch, logs['acc'],batch, logs['val_acc'])))
callback_list = [checkpoint, batch_print_callback]


history = model.fit(X_train,Y_train,batch_size=512,epochs=150,validation_data=(X_valid,Y_valid),callbacks = callback_list,shuffle = True)

model.summary()


# print the curve of acc and val_acc
'''plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
#plt.show()
plt.savefig(outputfile+"/plot-accuracy.png")'''

print(">>>>> Finish Training !!!")
