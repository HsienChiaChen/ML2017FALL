import numpy as np
import sys
import os  #mode = 3
from gensim.models import word2vec
import jieba

#str1 = "provideData/training_data/"
str1 = sys.argv[1]
str2 = "_train.txt"
train_data = []
for i in range(5):
	str4 = str(i+1)+str2
	str3 = os.path.join(str1,str4)
	train_data.append(str3)
#jiebaDict = "dict.txt.big"
jiebaDict = sys.argv[2]
stop_file = sys.argv[3]

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
stopwords = [line.rstrip() for line in open(stop_file,'r')]

for i in range(len(train)):
	train[i] = list(jieba.cut(train[i],cut_all = False))
	# add stopword
	if True:
		j = 0
		while j != len(train[i]):
			if train[i][j] in stopwords:
				train[i].remove(train[i][j])
				j -= 1
			j += 1

outputfile = open("corpus.txt",'w')
sentence_unit = 1
for i in range(len(train)):
	if i > 1:
		for j in range(len(train[i-2])):
			outputfile.write(train[i-2][j]+' ')
	if i != 0:
		for j in range(len(train[i-1])):
			outputfile.write(train[i-1][j]+' ')
	for j in range(len(train[i])):
		outputfile.write(train[i][j]+' ')
	outputfile.write('\n')
outputfile.close()
print("Corpus Output Successfully")

sentences = word2vec.Text8Corpus("corpus.txt")
model = word2vec.Word2Vec(sentences, size = 100,min_count=5,window = 7,sg = 1)
model.save("word_vector.model.bin")
