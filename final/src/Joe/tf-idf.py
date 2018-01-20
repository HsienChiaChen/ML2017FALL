import numpy as np
import sys
import csv
import gensim
from gensim.models import word2vec
import jieba
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer


# ================= #
# File Path Setting #
# ================= #
str1 = "provideData/training_data/"
str2 = "_train.txt"
train_data = []
for i in range(5):
	str3 = str1+str(i+1)+str2
	train_data.append(str3)
test_data     = "provideData/testing_data.csv" 
model_file = "w2v.bin"
jiebaDict = "dict.txt"


# ===================== #
# Loading Training Data #
# ===================== #

testq = []
testa = []

def load_data():
	print(">>>>> Loading Data ...")
	print("   >>> Testing Data")
	rownum = 0
	with open(test_data,'r',encoding = 'UTF-8') as f:
		l = f.readline()
		while l:
			if rownum != 0:
				l0 = l.strip('\n').split(',',maxsplit = 2)
				testq.append(l0[1].replace('A:','').replace('B:','').replace('C:','').replace('D:','').replace(' ','').replace('\t',''))
				testa.append(l0[2].replace('A:','').replace('B:','').replace('C:','').replace('D:','').replace(' ','').split("\t"))
			l = f.readline()
			rownum += 1


# =========================== #
# Text Sequence Preprocessing #
# =========================== #

def data_preprocessing():

	# Load Dictionary and Cut Words
	#jieba.set_dictionary(jiebaDict)
	for i in range(len(testq)):
		testq[i] = list(jieba.cut(testq[i],cut_all = False))
	for i in range(len(testa)):
		for j in range(len(testa[i])):
			testa[i][j] = list(jieba.cut(testa[i][j],cut_all = False))

	# Combine from list of string to a sentance with space
	for i in range(len(testq)):
		str1 = ""
		for j in range(len(testq[i])):
			str1 += testq[i][j]
			str1 += ' '
		testq[i] = str1
	for i in range(len(testa)):
		for j in range(len(testa[i])):
			str1 = ""
			for k in range(len(testa[i][j])):
				str1 += testa[i][j][k]
				str1 += ' '
			testa[i][j] = str1

# ================================================ #
# Calculate similartity between words in sentences #
# ================================================ #

def calsimilarity(k, vectorizer):

	# temp -> question string
	tfidf_q = vectorizer.transform([testq[k]]).toarray()
	temp = testq[k].split()
	sim = []
	# For each answer, compute similarity
	PAIR_AVG = False
	SENTENCE_AVG = True
	for choose in range(6):

		# ansstr -> answer string
		tfidf_a = vectorizer.transform([testa[k][choose]]).toarray()
		ansstr = testa[k][choose].split()
		oov_q = 0
		oov_a = 0
		sumsim = 0.0

		if PAIR_AVG:
			# get similarity from each pair of words (1 in Q, 1 in A)
			for i in range(len(temp)):
				if (temp[i] not in model.wv.vocab):
					oov_q += 1
			for i in range(len(ansstr)):
				if (ansstr[i] not in model.wv.vocab):
					oov_a += 1
			for i in range(len(temp)):
				if (temp[i] in model.wv.vocab):
					for j in range(len(ansstr)):
						# not in dict -> take it as oov.				
						if (ansstr[j] in model.wv.vocab):
							# in dictionary -> summation the similarity
							if((vectorizer.vocabulary_.get(temp[i]) != None) and (vectorizer.vocabulary_.get(ansstr[j]) != None)):
								sumsim += model.similarity(temp[i],ansstr[j])*tfidf_q[0][vectorizer.vocabulary_.get(temp[i])]*tfidf_a[0][vectorizer.vocabulary_.get(ansstr[j])]
	
			# take average
			sumsim /= (len(temp))*(len(ansstr))
			sim.append(sumsim)

		elif SENTENCE_AVG:
			q_vector = np.zeros(400, dtype='float32')
			a_vector = np.zeros(400, dtype='float32')
			for i in range(len(temp)):
				if (temp[i] in model.wv.vocab) and (vectorizer.vocabulary_.get(temp[i]) != None):
					q_vector += model.wv[temp[i]]*tfidf_q[0][vectorizer.vocabulary_.get(temp[i])]
				else:
					oov_q += 1
			for i in range(len(ansstr)):
				if (ansstr[i] in model.wv.vocab) and (vectorizer.vocabulary_.get(ansstr[i]) != None):
					a_vector += model.wv[ansstr[i]]*tfidf_a[0][vectorizer.vocabulary_.get(ansstr[i])]
				else:
					oov_a += 1
			'''if(len(temp) != oov_q and len(ansstr) != oov_a):
				q_vector /= len(temp)-oov_q
				a_vector /= len(ansstr)-oov_a'''
			if(len(temp) != oov_q):
				q_vector /= len(temp)-oov_q
			if(len(ansstr) != oov_a):
				a_vector /= len(ansstr)-oov_a
			#q_vector /= len(temp)
			#a_vector /= len(ansstr)
			sumsim = 1-spatial.distance.cosine(q_vector,a_vector)
			sim.append(sumsim)
				
	sim = np.array(sim)
	return(np.argmax(sim))

def predict_answer():
	file = open('corpus.txt', 'r', encoding = 'utf-8')
	corpus = file.read().split('\n')
	vectorizer = TfidfVectorizer()
	vectorizer.fit_transform(corpus)
	

	res = []
	for i in range(len(testq)):
		if i%1000 == 999:
			print(i+1)
		res.append(calsimilarity(i, vectorizer))

	# Output Predict File
	outputfile = open("predict_sentence.csv",'w')
	outputfile.write('id,ans\n')
	for i in range(len(res)):
		outputfile.write(str(i+1)+','+str(res[i])+'\n')
	outputfile.close()
	print("Output Successfully")

load_data()
data_preprocessing()
model = gensim.models.Word2Vec.load(model_file)
predict_answer()
