from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
import csv
import gensim
from gensim.models import word2vec
import pickle
import jieba
from scipy import spatial
from keras.preprocessing import sequence
import numpy as np
from random import shuffle

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

test_data = 'provideData/testing_data.csv'
w2v_model = gensim.models.Word2Vec.load('w2v.bin')

print(">>>>> Loading Data ...")
print(">>> Testing Data")
testq = []
testa = []
rownum = 0
with open(test_data,'r',encoding = 'utf-8') as f:
	l = f.readline()
	while l:
		if rownum != 0:
			l0 = l.strip('\n').split(',',maxsplit = 2)
			testq.append(l0[1].replace('A:','').replace('B:','').replace('C:','').replace('D:','').replace(' ','').replace('\t',''))
			testa.append(l0[2].replace('A:','').replace('B:','').replace('C:','').replace('D:','').replace(' ','').split('\t'))
		l = f.readline()
		rownum += 1

testq_vec = []
for i in range(len(testq)):
    vector = []
    testq[i] = list(jieba.cut(testq[i],cut_all = False))
    counter = 0
    for j in range(min(len(testq[i]),50)):
        if(testq[i][j] in w2v_model.wv.vocab):
            vector.append(w2v_model.wv[testq[i][j]])
    for _ in range(50-len(vector)):
        vector.append([0])
    vector = sequence.pad_sequences(vector, maxlen=400, padding='post', dtype='float32', value=0)
    testq_vec.append(vector)

testq_vec = np.stack(testq_vec, axis=0)

testa_vec = [[],[],[],[],[],[]]
for j in range(6):
    for i in range(len(testa)):
        vector = []
        testa[i][j] = list(jieba.cut(testa[i][j],cut_all = False))
        for word in testa[i][j]:
            if(word in w2v_model.wv.vocab):
                vector.append(w2v_model.wv[word])
            else:
                vector.append(np.zeros(400))
        for _ in range(50-len(vector)):
            vector.append([0])
        vector = sequence.pad_sequences(vector, maxlen=400, padding='post', dtype='float32', value=0)
        testa_vec[j].append(vector)
    testa_vec[j] = np.stack(testa_vec[j], axis=0)



latent_dim = 256
time_steps = 50
input_dim = 400
input_shape = (time_steps, input_dim)

inputs = Input(shape=input_shape)
encoded = LSTM(latent_dim)(inputs)

encoder = Model(inputs, encoded)

encoder.load_weights('encoder.h5')

represent_question = encoder.predict(testq_vec)
represent_answer = []
for i in range(len(testa_vec)):
	represent_answer.append(encoder.predict(testa_vec[i]))

outputfile = open('predict.csv','w')
outputfile.write('id,ans\n')
for i in range(len(represent_question)):
	opt = 0
	min_dist = float('inf')
	for j in range(6):
		dist = np.linalg.norm(represent_question[i]-represent_answer[j][i])
		if(min_dist > dist):
			min_dist = dist
			opt = j
	outputfile.write(str(i+1)+','+str(opt)+'\n')
outputfile.close()