import numpy as np
#import matplotlib.pyplot as plt
import sys
import csv
from math import floor
from random import shuffle
import os  #mode = 3

import keras
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Embedding, Activation, Input, Reshape
from keras.layers.merge import Dot, Add, Concatenate
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.regularizers import l2

# ========================= #
# Choose the Execution Mode #
# ========================= #

#import time
#st = time.time()

testcsv = sys.argv[1]
predictfile = sys.argv[2]
ID_to_user  = sys.argv[3]
ID_to_movie = sys.argv[4]
modelfile = sys.argv[5]
modelname = sys.argv[6:]
# ===================== #
# Loading Training Data #
# ===================== #

print(">>>>> Loading Data ...")

test = []
testfile = open(testcsv,'r')
rownum = 0
table = csv.reader(testfile, delimiter = ',')
for r in table:
	if rownum !=0:
		l = r[1:]
		test.append(list(map(int,l)))
	rownum+=1
testfile.close()

index1 = []
index2 = []
file1 = open(ID_to_user,'r')
table = csv.reader(file1,delimiter = ' ')
for r in table:
	index1.append(list(map(int,r)))
file1.close()
file2 = open(ID_to_movie,'r')
table = csv.reader(file2,delimiter = ' ')
for r in table:
	index2.append(list(map(int,r)))
file2.close()

index1 = np.array(index1)
index2 = np.array(index2)
index1 = index1.reshape(index1.shape[0],)
index2 = index2.reshape(index2.shape[0],)

user_to_ID = {key:ID for ID,key in enumerate(index1)}
movie_to_ID = {key:ID for ID,key in enumerate(index2)}


test = np.array(test)
for i in range(len(test)):
	test[i][0] = user_to_ID[test[i][0]]
	test[i][1] = movie_to_ID[test[i][1]]

def rmse(y_pred,y):
	return K.sqrt(K.mean((y_pred-y)**2))

# ================== #
# Predict the Answer #
# ================== #

print("\n>>>>> Load Model and Predict ...")
model = load_model(modelfile,custom_objects = {'rmse': rmse})
model.load_weights(modelfile)
result = model.predict([test[:,0],test[:,1]])
count = 1

for modelx in modelname[1:]:
	model = load_model(modelx,custom_objects = {'rmse': rmse})
	model.load_weights(modelx)
	result += model.predict([test[:,0],test[:,1]])
	count +=1

result = np.array(result)/count
NORMALIZE = False
RESCALE = False

if NORMALIZE:
	result = result*std + mean
if RESCALE:
	result = result*4+1

result[result>5.0]=5.0
result[result<1.0]=1.0
print(result[0:15])

outputfile = open(predictfile,'w')
outputfile.write('TestDataID,Rating\n')
for i in range(len(result)):
	outputfile.write(str(i+1)+','+str(result[i].squeeze())+'\n')
outputfile.close()
print("Output Successfully")


#end = time.time()
#print("Executing Time: "+str(int((end-st)//60))+" minute "+str((end-st)%60)+" second")

