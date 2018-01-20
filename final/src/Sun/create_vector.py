import numpy as np
import sys
import os  #mode = 3
from gensim.models import word2vec
import jieba


str1 = "provideData/training_data/"
str2 = "_train.txt"
train_data = []
for i in range(5):
    str3 = str1+str(i+1)+str2
    train_data.append(str3)
#test_data     = "provideData/testing_data.csv" 
#worddict      = "m200.model.bin"
jiebaDict = "dict.txt"


print(">>>>> Loading Data ...")

train = []

print("   >>> Training set")
for i in range(len(train_data)):
    with open(train_data[i],'r',encoding = 'UTF-8') as f:
        l = f.readline()
        while l:
            l = l.strip('\n')
            l = l.replace('\t','')
            l = l.replace(' ','')
            train.append(l)
            l = f.readline()

jieba.set_dictionary(jiebaDict)
for i in range(len(train)):
    train[i] = list(jieba.cut(train[i],cut_all = False))

outputfile = open("corpus.txt","w")
for i in range(len(train)) :
    for j in range(len(train[i])) :
        outputfile.write(train[i][j] + ' ')
    outputfile.write('\n')

outputfile = open("haodnosi.txt",'w')
for i in range(len(train)):
    if i > 1:
        for j in range(len(train[i - 2])) :
            outputfile.write(train[i - 2][j] + ' ')
        for j in range(len(train[i - 1])) :
            outputfile.write(train[i - 1][j] + ' ')
        for j in range(len(train[i])):
    	    outputfile.write(train[i][j] + ' ')
        outputfile.write('\n')
outputfile.close()

outputfile = open("fendonsi.txt","w")
for i in range(len(train)) :
    rand = np.random.randint(len(train), size = 2)
    if abs(rand[0] - rand[1]) > 3:
        M = max(rand[0],rand[1])
        m = min(rand[0],rand[1])
        for j in range(len(train[m])) :
            outputfile.write(train[m][j] + ' ')
        for j in range(len(train[m + 1])) :
            outputfile.write(train[m + 1][j] + ' ')
        for j in range(len(train[M - 1])) :
            outputfile.write(train[M - 1][j] + ' ')
        for j in range(len(train[M])) :
            outputfile.write(train[M][j] + ' ')
        outputfile.write('\n')
outputfile.close()

print("Corpus Output Successfully")

sentences = word2vec.Text8Corpus("corpus.txt")
model = word2vec.Word2Vec(sentences, size = 200,min_count=5,window = 10)

model.save("m200chinese.model.bin")
