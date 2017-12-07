#from __future__ import print_function  # mode = 3
import numpy as np
#import matplotlib.pyplot as plt
import sys
import csv
from math import floor
from random import shuffle
import os  #mode = 3
import gensim
from gensim.models import word2vec
import _pickle as pk

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Activation, LSTM, MaxPooling1D, Conv1D, GRU
from keras.callbacks import ModelCheckpoint, LambdaCallback
# ========================= #
# Choose the Execution Mode #
# ========================= #

'''mode = 2
if mode == 1:
	path = os.environ.get("GRAPE_DATASET_DIR")
	print(path)
	train_label   = os.path.join(path,"training_label.txt")
	train_nolabel = os.path.join(path,"training_nolabel.txt")
	test_data     = os.path.join(path,"testing_data.txt")
	worddict      = os.path.join(path,"m200.model.bin")
	outputfile = "output"
elif mode == 2:
	train_label   = "data/training_label.txt"
	train_nolabel = "data/training_nolabel.txt"
	test_data     = "data/testing_data.txt" 
	worddict      = "m200.model.bin"
	outputfile = "output12GRU200dim"
else:'''
train_label   = sys.argv[1]
train_nolabel = sys.argv[2]
#test_data     = sys.argv[3]
dictfile      = sys.argv[3]
worddict      = sys.argv[4]
outputfile    = sys.argv[5]

# ===================== #
# Loading Training Data #
# ===================== #

print(">>>>> Loading Data ...")

trainA = []
label = []
trainB = []
test = []
print("   >>> Training set with label")
with open(train_label,'r',encoding = 'UTF-8') as f:
	l = f.readline()
	while l:
		line = l.strip('\r\n').split(" +++$+++ ")
		label.append(int(line[0]))
		trainA.append(line[1])
		l = f.readline()
print("   >>> Training set without label")
with open(train_nolabel,'r',encoding = 'UTF-8') as f:
	l = f.readline()
	while l:
		trainB.append(l.strip('\r\n'))
		l = f.readline()
'''print("   >>> Testing Data")
rownum = 0
with open(test_data,'r',encoding = 'UTF-8') as f:
	l = f.readline()
	while l:
		if rownum != 0: 
			test.append(l.strip('\r\n').split(',',maxsplit=1)[1])
		l = f.readline()
		rownum += 1'''
# =========================== #
# Text Sequence Preprocessing #
# =========================== #

print(">>>>> Text Preprocessing ...")

method = 2
if method == 1: # Bag of words

	'''token = Tokenizer(num_words = 500)
	token.fit_on_texts(trainA)
	token.fit_on_texts(trainB)
	token.fit_on_texts(test)

	trainA = token.texts_to_matrix(trainA,'binary')
	trainB = token.texts_to_matrix(trainB,'binary')
	test   = token.texts_to_matrix(test,'binary')
	
	trainA = np.array(trainA)
	trainB = np.array(trainB)
	label = np.array(label)
	test = np.array(test)'''

elif method == 2: # Word Embedding with gensim

	#token = Tokenizer(num_words = 13200,filters = ' ')
	#token.fit_on_texts(trainA)
	#token.fit_on_texts(trainB)
	#token.fit_on_texts(test)
	#pk.dump(token,open("dict",'wb'))
	#print("finish store dict!!!")
	token = pk.load(open(dictfile,'rb'))
	print("finish load dict!!!")


	trainA = token.texts_to_sequences(trainA)
	trainB = token.texts_to_sequences(trainB)
	#test   = token.texts_to_sequences(test)

	trainA = np.array(trainA)
	trainB = np.array(trainB)
	label  = np.array(label)
	#test   = np.array(test)

	vec = word2vec.Word2Vec.load(worddict)
	word2idx = {"_PAD":0}
	vocab_list = [(k,vec.wv[k]) for k,v in vec.wv.vocab.items()]
	embeddings_matrix = np.zeros((len(vec.wv.vocab.items())+1,vec.vector_size))
	for i in range(len(vocab_list)):
		word = vocab_list[i][0]
		word2idx[word] = i+1
		embeddings_matrix[i+1] = vocab_list[i][1]
	EMBEDDING_DIM = vec.vector_size
	print(embeddings_matrix.shape)

	maxlength = 30
	trainA = pad_sequences(trainA,maxlen = maxlength)
	trainB = pad_sequences(trainB,maxlen = maxlength)
	#test   = pad_sequences(test  ,maxlen = maxlength)

# ==================================== #
# Implement Split and Shuffle Function #
# ==================================== #

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

percentage = 0.8 # training set portion
X_train,Y_train,X_valid,Y_valid = split_valid_set(trainA,label,percentage)

print("Size of training set: "+str(X_train.shape))
print("Size of validation  : "+str(X_valid.shape))

# ======================== #
# Build RNN Neural Network #
# ======================== #


print("\n>>>>> Build RNN Neural Network ...")

model = Sequential()
if method == 2:

	model.add(Embedding(len(embeddings_matrix),EMBEDDING_DIM,weights = [embeddings_matrix],input_length = maxlength,trainable = True))
	#model.add(Conv1D(128,10,padding = 'same',activation = 'relu'))
	model.add(LSTM(300,dropout = 0.3,return_sequences = False))
	#model.add(LSTM(300,dropout = 0.3,go_backwards = True))
	#model.add(GRU(128,activation = 'tanh',return_sequences = True,dropout = 0.3))
	#model.add(GRU(128,activation = 'tanh',return_sequences = True,dropout = 0.3))
	#model.add(GRU(128,activation = 'tanh',return_sequences = True,dropout = 0.3))
	#model.add(GRU(1000,activation = 'tanh',return_sequences = False,dropout = 0.3))
	
	model.add(Dense(units = 300, activation = 'relu'))
	#model.add(Dropout(0.3))
	#model.add(Dense(units = 64, activation = 'relu'))
	#model.add(Dense(units = 16, activation = 'relu'))
	model.add(Dense(units = 1, activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])

elif method == 1:

	model.add(Dense(input_dim = 500,units = 256,activation = 'relu'))
	model.add(Dropout(0.3))
	model.add(Dense(units = 128,activation = 'relu'))
	model.add(Dropout(0.3))
	model.add(Dense(units = 64, activation = 'relu'))
	model.add(Dropout(0.3))
	model.add(Dense(units = 32,activation = 'relu'))
	model.add(Dense(units = 16,activation = 'relu'))
	model.add(Dense(units = 1, activation = 'sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


if not os.path.exists(outputfile):
	os.makedirs(outputfile)
filepath = outputfile + "/model-{val_acc:.5f}.h5"

checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc',verbose = 1,save_best_only=True,mode='max') 
batch_print_callback = LambdaCallback(on_epoch_end=lambda batch, logs:
						print('\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %
						(batch, logs['acc'],batch, logs['val_acc'])))
#callback_list = [checkpoint, batch_print_callback]
callback_list = [checkpoint]

history = model.fit(X_train,Y_train,batch_size=128,epochs=7,validation_data=(X_valid,Y_valid),callbacks = callback_list,shuffle = True)

'''k = 1
while len(trainB) > 0:
	train(X_train,Y_train,50)
	print("   >>> (1/3) Predict No.%d ..." % k)
	temp_result = model.predict(trainB)
	print("   >>> (2/3) Take the high predict probability data ....")
	one_index  = np.argwhere(temp_result>0.8).flatten()
	zero_index = np.argwhere(temp_result<0.2).flatten()

	#one_seq	   = trainB[one_index]
	one_label  = np.ones(len(one_index))
	#zero_seq   = trainB[zero_index]
	zero_label = np.zeros(len(zero_index))
	
	#app_seq   = np.concatenate((trainB[one_index],trainB[zero_index]),axis = 0)
	#app_label = np.concatenate((one_label,zero_label),axis = 0)
	#del_index = np.concatenate((one_index,zero_index),axis = 0)

	print("   >>> (3/3) Concatenate and Delete")
	X_train = np.concatenate((X_train,trainB[one_index],trainB[zero_index]),axis = 0)
	Y_train = np.concatenate((Y_train,one_label,zero_label),axis = 0)
	trainB = np.delete(trainB,np.concatenate((one_index,zero_index),axis = 0),axis = 0)
	
	#print("X_train's shape: "+str(X_train.shape))
	#print("Y_train's shape: "+str(Y_train.shape))
	print("Remain Unlabel data: "+str(trainB.shape[0]))
	k += 1
#train(X_train,Y_train)
'''

# ================== #
# Predict the answer #
# ================== #
'''
result = model.predict(test)
print(result[0:15])
res = []
for i in range(len(test)):
	if result[i]>=0.5:  ans = 1
	else: ans=0
	res.append(ans)

outputfile = open("predict.csv",'w')
outputfile.write('id,label\n')
for i in range(len(res)):
	outputfile.write(str(i)+','+str(res[i])+'\n')
outputfile.close()
print("Output Successfully")
'''

# ================================== #
# print the curve of acc and val_acc #
# ================================== #
'''plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
#plt.show()
plt.savefig(outputfile+"/plot-accuracy.png")'''

