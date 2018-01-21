import numpy as np
import sys
import csv
import gensim
from gensim.models import word2vec
import jieba
from scipy import spatial

# ================= #
# File Path Setting #
# ================= #

test_data = sys.argv[1]
model_file = sys.argv[2]
jiebaDict = sys.argv[3]
stop_file = sys.argv[4]
predict_file = sys.argv[5]

# ===================== #
# Loading Training Data #
# ===================== #

testq = []
testa = []

stopwords = [line.rstrip() for line in open(stop_file,'r')]

def load_data():
	print(">>>>> Loading Data ...")
	print("   >>> Testing Data")
	rownum = 0
	with open(test_data,'r',encoding = 'UTF-8') as f:
		l = f.readline()
		while l:
			if rownum != 0:
				l0 = l.strip('\n').split(',',maxsplit = 2)
				testq.append(l0[1].replace('A:','').replace('B:','').replace('C:','').replace(' ','').replace('\t',''))
				testa.append(l0[2].replace('A:','').replace('B:','').replace('C:','').replace(' ','').split("\t"))
			l = f.readline()
			rownum += 1


# =========================== #
# Text Sequence Preprocessing #
# =========================== #

def data_preprocessing():

	# Load Dictionary and Cut Words
	STOP = True
	jieba.set_dictionary(jiebaDict)
	for i in range(len(testq)):
		testq[i] = list(jieba.cut(testq[i],cut_all = False))
		if STOP:
			j = 0
			while j != len(testq[i]):
				if testq[i][j] in stopwords:
					testq[i].remove(testq[i][j])
					j-=1
				j+=1

	for i in range(len(testa)):
		for j in range(len(testa[i])):
			testa[i][j] = list(jieba.cut(testa[i][j],cut_all = False))
			if STOP:
				k = 0
				while k != len(testa[i][j]):
					if testa[i][j][k] in stopwords:
						testa[i][j].remove(testa[i][j][k])
						k-=1
					k+=1


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


def calsimilarity(k):

	# temp -> question string
	temp = testq[k].split()
	sim = []

	# For each answer, compute similarity
	for choose in range(6):

		# ansstr -> answer string
		ansstr = testa[k][choose].split()
		oov = 0
		sumsim = 0.0

		len_a = len(ansstr)
		ansstr = [w for w in ansstr if w in model.wv.vocab]
		temp = [w for w in temp if w in model.wv.vocab]
		for i in range(len(temp)):
			for j in range(len(ansstr)):
				prob = model.similarity(temp[i],ansstr[j])
				if prob > 0.2:
					sumsim += prob
		if len_a > 0:
			sumsim /= float(len_a)
		else:
			sumsim = 0.0
		sim.append(sumsim)

	sim = np.array(sim)
	return(np.argmax(sim))

def predict_answer():

	# Calculate the similarity between sentences
	res = []
	for i in range(len(testq)):
		if i%1000 == 999:
			print(i+1)
		res.append(calsimilarity(i))

	# Output Predict File
	outputfile = open(predict_file,'w')
	outputfile.write('id,ans\n')
	for i in range(len(res)):
		outputfile.write(str(i+1)+','+str(res[i])+'\n')
	outputfile.close()
	print("Output Successfully")

load_data()
data_preprocessing()
model = gensim.models.Word2Vec.load(model_file)
predict_answer()

