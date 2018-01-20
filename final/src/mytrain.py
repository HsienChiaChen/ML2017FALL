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
'''str1 = "../provideData/training_data/"
str2 = "_train.txt"
train_data = []
for i in range(5):
	str3 = str1+str(i+1)+str2
	train_data.append(str3)'''
#test_data     = "../provideData/testing_data.csv" 
#model_file = "m0116chinese.model.bin"
#jiebaDict = "dict.txt.big"
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
	#print()
	#print(testq[k].replace(" ",""))
	# For each answer, compute similarity
	PAIR_AVG = True
	PAIR_AVG_N = False
	SENT_AVG = False
	for choose in range(6):

		# ansstr -> answer string
		ansstr = testa[k][choose].split()
		oov = 0
		sumsim = 0.0

		if PAIR_AVG:
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
		elif PAIR_AVG_N:
			# get similarity from each pair of words (1 in Q, 1 in A)
			len_a = len(ansstr)
			ansstr = [w for w in ansstr if w in model.wv.vocab]
			temp = [w for w in temp if w in model.wv.vocab]
			ansstr = [w for w in ansstr if w not in temp]
			try:
				sumsim = model.n_similarity(temp,ansstr)
			except:
				sumsim = 0.0
			sim.append(sumsim)
			#print(testa[k][choose],sumsim)
		elif SENT_AVG:
			len_a = len(ansstr)
			ansstr = [w for w in ansstr if w in model.wv.vocab]
			temp = [w for w in temp if w in model.wv.vocab]
			v_t = np.zeros((100,))
			v_a = v_t
			for w in temp:
				v_t += model.wv[w]
			for w in ansstr:
				v_a += model.wv[w]
			if len(temp) > 0:
				v_t /= float(len(temp))
			else:
				v_t = np.zeros((100,))
			if len(ansstr) > 0:
				v_a /= float(len(ansstr))	
			else:
				v_a = np.zeros((100,))
			if np.linalg.norm(v_t) != 0.0 and np.linalg.norm(v_a) != 0.0:
				sumsim = 1-spatial.distance.cosine(v_t,v_a)
			else:
				sumsim = 0.0
			sim.append(sumsim)

	sim = np.array(sim)
	#print(sim)
	#print("=======> "+str(testa[k][np.argmax(sim)].replace(" ","")))
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

'''for i in range(len(testq)):
	l = testq[i].split()
	for j in l:
		if j not in model.wv.vocab:
			testq[i] = testq[i].replace(j,'')
			s = ""
			for k in range(len(j)):
				s += (j[k] + " ")
			testq[i] = testq[i] + s

for i in range(len(testa)):
	for j in range(len(testa[i])):
		l = testa[i][j].split()
		for k in l:
			if k not in model.wv.vocab:
				testa[i][j] = testa[i][j].replace(k,'')
				s = ""
				for kk in range(len(k)):
					s += (k[kk] + " ")
				testa[i][j] = testa[i][j] + s'''

predict_answer()

