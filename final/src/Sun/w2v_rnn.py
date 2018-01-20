import numpy as np
import sys
import os
import string
import sklearn
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence,text
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn import cross_validation

sl = 30
vl = 200

model_w2v = Word2Vec.load('m200chinese.model.bin') 

hao = open('haodnosi.txt','r')
fen = open('fendonsi.txt','r')

n_hao = 378500
n_fen = 378500

len_hao = 756998
len_fen = 757000

#read corpus
X_hao = []
Y_train = []
X_fen = []
#haodonsi
for i in range(len_hao) :
    line = hao.readline()
    line = line.strip('\n')
    line = line.split(' ')
    X_hao.append(line)

for i in range(len_fen) :
    line = fen.readline()
    line = line.strip('\n')
    line = line.split(' ')
    X_fen.append(line)

X_trainarr = np.zeros((n_hao + n_fen,sl,vl))

for i in range(n_hao) :
    k = 0
    pos = np.random.randint(len_hao,size=1)
    for word in X_hao[pos[0]] :
        if k > sl - 1 :
            break
        if word in model_w2v.wv.vocab:
            X_trainarr[i][k] = model_w2v[word]
            k += 1

for i in range(n_fen) :
    k = 0
    pos = np.random.randint(len_fen,size=1)
    for word in X_fen[pos[0]] :
        if k > sl - 1 :
            break
        if word in model_w2v.wv.vocab:
            X_trainarr[i + n_hao][k] = model_w2v[word]
            k += 1

for i in range(n_hao) :
    Y_train.append(1)

for i in range(n_fen) :
    Y_train.append(0)

X_tr, X_val, Y_tr, Y_val = cross_validation.train_test_split(X_trainarr,Y_train,test_size=0.1)

model = Sequential()

model.add(LSTM(100,input_shape=(sl,vl)))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

ADAM = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy',optimizer=ADAM,metrics=['accuracy'])

logger = CSVLogger('rnn.csv')

model.fit(X_tr,Y_tr,batch_size=512,epochs=15,validation_data=(X_val,Y_val),callbacks=[logger])

model.save('rnn.h5')

prediction = model.predict(X_val)
accpn = 0
accpp = 0
accnp = 0
accnn = 0
for i in range(len(X_val)) :
    if Y_val[i] == 0 :
        if prediction[i] < 0.5 :
            accnn += 1
        else :
            accnp += 1
    else :
        if prediction[i] < 0.5:
            accpn += 1
        else :
            accpp += 1
print(accpp,accpn,accnn,accnp) 
