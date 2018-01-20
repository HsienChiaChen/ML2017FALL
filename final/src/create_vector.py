import numpy as np
import sys
import os  #mode = 3
from gensim.models import word2vec
import jieba
import pyemd

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
#word_vectors = model.wv
#word_vectors.save("W2V")
#from gensim.models.keyedvectors import KeyedVectors
#word_vectors = KeyedVectors.load("W2V")
#print(word_vectors.doesnt_match(u'醫院 病人 護士 慶祝'.split()))


'''model = word2vec.Word2Vec.load("m0116chinese.model.bin")
sentence_1 = u'媽媽 煮飯'.split()
sentence_2 = u'吃飯 雞湯'.split()
sentence_3 = u'醫生 看病'.split()
distance12 = model.wmdistance(sentence_1,sentence_2)
distance13 = model.wmdistance(sentence_1,sentence_3)
print(distance12)
print(distance13)




print("========================")
print(model.n_similarity(u'上課 讀書'.split(),u'上課 睡覺'.split()))
print(model.n_similarity(u'上課 讀書'.split(),u'回家 吃飯'.split()))


#model.save("m0116chinese.model.bin")
'''
