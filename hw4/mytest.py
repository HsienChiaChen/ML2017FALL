#from __future__ import print_function  # mode = 3
import numpy as np
#import matplotlib.pyplot as plt
import sys
import csv
from math import floor
from random import shuffle
import os  #mode = 3
import gensim
import _pickle as pk

import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Activation, LSTM, MaxPooling1D
from keras.callbacks import ModelCheckpoint, LambdaCallback
# ========================= #
# Choose the Execution Mode #
# ========================= #


#import time
#st = time.time()

'''mode = 2
if mode == 1:
	path = os.environ.get("GRAPE_DATASET_DIR")
	print(path)
	traincsv = os.path.join(path,"train.csv")
	testcsv = os.path.join(path,"test.csv")
	outputfile = "output"
elif mode == 2:
	train_label   = "data/training_label.txt"
	train_nolabel = "data/training_nolabel.txt"
	test_data     = "data/testing_data.txt" 
	modelfile1  = sys.argv[1]
	#modelfile2 = sys.argv[2]
	#modelfile3 = sys.argv[3]
	#modelfile4 = sys.argv[4]
else:'''
#train_label = sys.argv[1]
#train_nolabel = sys.argv[2]
test_data = sys.argv[1]
predictfile = sys.argv[2]
dictfile = sys.argv[3]
modelfile1 = sys.argv[4]
modelfile2 = sys.argv[5]
modelfile3 = sys.argv[6]
modelfile4 = sys.argv[7]

# ===================== #
# Loading Training Data #
# ===================== #

print(">>>>> Loading Data ...")

trainA = []
label = []
trainB = []
test = []
'''print("   >>> Training set with label")
with open(train_label) as f:
	l = f.readline()
	while l:
		line = l.strip('\r\n').split(" +++$+++ ")
		label.append(int(line[0]))
		trainA.append(line[1])
		l = f.readline()
print("   >>> Training set without label")
with open(train_nolabel) as f:
	l = f.readline()
	while l:
		trainB.append(l.strip('\r\n'))
		l = f.readline()'''
print("   >>> Testing Data")
rownum = 0
with open(test_data) as f:
	l = f.readline()
	while l:
		if rownum != 0: 
			test.append(l.strip('\r\n').split(',',maxsplit=1)[1])
		l = f.readline()
		rownum += 1
# =========================== #
# Text Sequence Preprocessing #
# =========================== #

print(">>>>> Text Preprocessing ...")

method = 2
if method == 1: # Bag of Words

	'''token = Tokenizer(num_words = 500)
	token.fit_on_texts(trainA)
	token.fit_on_texts(trainB)
	token.fit_on_texts(test)

	#trainA = token.texts_to_matrix(trainA,'binary')
	#trainB = token.texts_to_matrix(trainB,'binary')
	test   = token.texts_to_matrix(test,'binary')
	
	#trainA = np.array(trainA)
	#trainB = np.array(trainB)
	#label = np.array(label)
	test = np.array(test)'''

elif method == 2: # Embeddings

	#token = Tokenizer(num_words = 13200)
	#token = Tokenizer(num_words = 13200,filters = ' ')
	#token.fit_on_texts(trainA)
	#token.fit_on_texts(trainB)
	#token.fit_on_texts(test)
	token = pk.load(open(dictfile,'rb'))
	print("finish load dict!!!")

	#trainA = token.texts_to_sequences(trainA)
	#trainB = token.texts_to_sequences(trainB)
	test   = token.texts_to_sequences(test)

	#trainA = np.array(trainA)
	#trainB = np.array(trainB)
	#label  = np.array(label)
	test   = np.array(test)

	maxlength = 30
	#trainA = pad_sequences(trainA,maxlen = maxlength)
	#trainB = pad_sequences(trainB,maxlen = maxlength)
	test   = pad_sequences(test  ,maxlen = maxlength)



# ================== #
# Predict the Answer #
# ================== #

print("\n>>>>> Load Model and Predict ...")
print("model 1")
model1 = load_model(modelfile1)
model1.load_weights(modelfile1)
model1.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics=['accuracy'])
#model1.summary()
result1 = model1.predict(test)

print("model 2")
model2 = load_model(modelfile2)
model2.load_weights(modelfile2)
model2.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics=['accuracy'])
#model2.summary()
result2 = model2.predict(test)

print("model 3")
model3 = load_model(modelfile3)
model3.load_weights(modelfile3)
model3.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics=['accuracy'])
#model3.summary()
result3 = model3.predict(test)

print("model 4")
model4 = load_model(modelfile4)
model4.load_weights(modelfile4)
model4.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics=['accuracy'])
#model4.summary()
result4 = model4.predict(test)

result = (result1 + result2 + result3 + result4) * 0.25
#result = (result1 + result2 + result3) / 3
#result = (result1 + result2) * 0.5
#result = result1

print(result[0:15])
res = []
for i in range(len(test)):
	if result[i]>=0.5:  ans = 1
	else: ans=0
	res.append(ans)

outputfile = open(predictfile,'w')
outputfile.write('id,label\n')
for i in range(len(res)):
	outputfile.write(str(i)+','+str(res[i])+'\n')
outputfile.close()
print("Output Successfully")


#end = time.time()
#print("Executing Time: "+str(int((end-st)//60))+" minute "+str((end-st)%60)+" second")

'''
outputfile = open("prob.csv",'w')
outputfile.write('id,model 1,model 2,total\n')
for i in range(len(res)):
	outputfile.write(str(i)+','+str(result1[i])+','+str(result2[i])+','+str(result3[i])+','+str(result4[i])+','+str(result[i])+'\n')
outputfile.close()
'''
