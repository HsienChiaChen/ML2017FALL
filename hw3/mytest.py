import numpy as np
import sys
import csv
from math import floor
from random import shuffle
import os

# plot
#from sklearn.metrics import confusion_matrix
#from utils import *
#import itertools


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.models import load_model
from keras.utils.vis_utils import plot_model


# Working on different workstation
#mode = 2

'''if mode == 1:
	path = os.environ.get("GRAPE_DATASET_DIR")
	traincsv = os.path.join(path,"train.csv")
	testcsv = os.path.join(path,"test.csv")
	outputfile = "output"
	modelfile = os.path.join(path,"save_models/model-0.63384.h5")
elif mode == 2:
	path = os.environ.get("HOME")
	path = path + '/Desktop/ML/hw3'
	traincsv = os.path.join(path,"train.csv")
	testcsv = os.path.join(path,"test.csv")
	outputfile = os.path.join(path,"predict_combine.csv")
	
	models = []
	#models.append('save_models/model-0.62731.h5')  #a
	models.append('save_models/model-0.62792.h5')  #a
	models.append('save_models/model-0.62898.h5')
	#models.append('save_models/model-0.63062.h5') #b
	#models.append('save_models/model-0.63192.h5') #b
	#models.append('save_models/model-0.63332.h5') #b
	#models.append('save_models/model-0.63384.h5') #b
	#models.append('save_models/model-0.63393.h5') #b
	models.append('save_models/model-0.63715.h5') #b
	#models.append('save_models_2/model-0.62304.h5')
	#models.append('save_models_2/model-0.62748.h5')
	#models.append('save_models_2/model-0.62757.h5')
	#models.append('save_models_2/model-0.62879.h5')
	#models.append('save_models_2/model-0.63166.h5')
	#models.append('save_models_2/model-0.63523.h5')
	models.append('save_models_2/model-0.63889.h5')
	#models.append('save_models_2/model-0.63932.h5')
	#models.append('save_models_3/model-0.61869.h5')
	#models.append('save_models_3/model-0.61668.h5')
	#models.append('save_models_3/model-0.61425.h5')
else:
	traincsv = sys.argv[1]
	testcsv = sys.argv[2]
	outputfile = sys.argv[3]
	modelfile = sys.argv[4] # import model file!!!'''
testcsv = sys.argv[1]
outputfile = sys.argv[2]
model1 = sys.argv[3]
model2 = sys.argv[4]
model3 = sys.argv[5]
model4 = sys.argv[6]
models = []
models.append(model1)
models.append(model2)
models.append(model3)
models.append(model4)


print(">>>>> Loading Testing Data ...")

# loading testing data

testcsv = open(testcsv,'r')
temp = csv.reader(testcsv,delimiter = ',')
test = []
rownum = 0
for r in temp:
	if rownum!=0:
		test.append(r[1].split())
	rownum+=1
testcsv.close()
test = np.array(test)
test = test.astype(float)

print("Size of testing  set: "+str(test.shape))

print("\n>>>>> Normalize the Data ...")
test  /= 255
test = test.reshape(len(test),48,48,1)

print("\n>>>>> Loading Model ...")

result = np.zeros((len(test),7))
i=1
for modelpath in models:
	print(">>>>> Predict Model No." + str(i))
	r = modelpath.split('.')
	weight = float(r[1])*0.0001
	if weight > 6.33:
		weight = weight*1.2/6
	elif weight < 6.28:
		weight = weight*0.8/6
	else:
		weight = weight*1.0/6
	print("weight = "+ str(weight))
	#modelfile = os.path.join(path,modelpath)
	modelfile = modelpath
	model = load_model(modelfile)
	model.load_weights(modelfile)
	model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics=['accuracy'])
	result += model.predict(test) * weight
	i+=1


#print("\n>>>>> Model Summary ...")
#model.summary()
#plot_model(model,to_file = 'model.png')


#print("\n>>>>> Predict test.csv ...")

#print(np.array(result).shape)

ans = []
for i in range(len(result)):
	prob = -1.0
	idx = -1
	for j in range(7):
		if result[i][j]>prob:
			prob = result[i][j]
			idx = j
	ans.append(idx)

print("\n>>>>> Output Predict File ...")
outputfile = open(outputfile,'w')
outputfile.write('id,label\n')
for i in range(len(ans)):
	outputfile.write(str(i)+','+str(ans[i])+'\n')
outputfile.close()
print("Output CSV File Successfully !!!")

'''print("======Plot confusion matrix======")

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


print("\n>>>>> Normalize the Data ...")
train /= 255
#test  /= 255

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
#test    = test.reshape(len(test),48,48,1)

Y = []
for i in range(len(Y_valid)):
	for j in range(len(Y_valid[i])):
		if Y_valid[i][j] == 1:
			Y.append(j)
			break
Y_valid = np.array(Y)
print("Size of training set: "+str(X_train.shape))
print("Size of validation  : "+str(X_valid.shape))
 
testresult = model.predict(X_valid)

ans = []
for i in range(len(testresult)):
	prob = -1.0
	idx = -1
	for j in range(7):
		if testresult[i][j]>prob:
			prob = testresult[i][j]
			idx = j
	ans.append(idx)

mat = confusion_matrix(Y_valid,ans)
label=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
np.set_printoptions(precision=2)
mat = mat.astype('float')/mat.sum(axis=1)[:,np.newaxis]
print(mat)
plt.figure()
plt.imshow(mat,interpolation = 'nearest',cmap = plt.cm.jet)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(7)
plt.xticks(tick_marks,label)
plt.yticks(tick_marks,label)
fmt = '.2f'
thresh = mat.max()/2.
for i,j in itertools.product(range(mat.shape[0]),range(mat.shape[1])):
	plt.text(j,i,format(mat[i,j],fmt),horizontalalignment='center',color='white' if mat[i,j] > thresh else 'black')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predict Label')
plt.show()
'''
