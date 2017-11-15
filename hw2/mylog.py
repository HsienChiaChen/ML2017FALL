import numpy as np
import csv
from math import log,floor
import pandas as pd
import sys

# ------------------------
# Read data from train.csv
# ------------------------

argtrainx = sys.argv[1] # X_train
argtrainy = sys.argv[2] # Y_train
argtest = sys.argv[3] # X_test
argout = sys.argv[4] # outputfile

print("----- Read the data -----")

'''data = []
rownum = 0
inputfile = open('X_train','r',encoding = 'big5')
table = csv.reader(inputfile, delimiter = ',')
for r in table:
	if rownum !=0:
		data.append([])
		for i in range(len(r)):
			data[rownum-1].append(float(r[i]))
	rownum+=1
inputfile.close()

label = []
rownum = 0
inputfile = open('Y_train','r',encoding = 'big5')
table = csv.reader(inputfile, delimiter = ',')
for r in table:
	if rownum!=0:
		label.append([])
		for i in range(len(r)):
			#label[rownum-1].append(1-int(r[i]))
			label[rownum-1].append(int(r[i]))
	rownum+=1
inputfile.close()
label = np.array(label)

testdata = []
rownum = 0
inputfile = open('X_test','r',encoding = 'big5')
table = csv.reader(inputfile, delimiter = ',')
for r in table:
	if rownum != 0:
		testdata.append([])
		for i in range(len(r)):
			testdata[rownum-1].append(float(r[i]))
	rownum+=1
inputfile.close()'''

data     = np.array(pd.read_csv(argtrainx,sep=',',header=0).values)
label    = np.array(pd.read_csv(argtrainy,sep=',',header=0).values)
testdata = np.array(pd.read_csv(argtest  ,sep=',',header=0).values)


data = np.array(data)
label = np.array(label)
testdata = np.array(testdata)

print(data.shape)
print(label.shape)
print(testdata.shape)

def _shuffle(X,Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize],Y[randomize])

print("----- Normalize the data -----")

alldata = np.concatenate((data,testdata))
mean = np.average(alldata,axis = 0)
sigma = np.std(alldata,axis = 0)
mean = np.tile(mean,(len(alldata),1))
sigma = np.tile(sigma,(len(alldata),1))
normed_data = (alldata-mean) / sigma

data = normed_data[:len(data)]
testdata = normed_data[len(data):]

def split_valid_data(data,label,percentage):
	N_train = int(floor(len(data)*percentage))
	X_train = data[:N_train]
	Y_train = label[:N_train]
	X_valid = data[N_train:]
	Y_valid = label[N_train:]
	return X_train, Y_train, X_valid, Y_valid
	
def sigmoid(z):
	res = 1/ (1.0+np.exp(-z))
	return np.clip(res,1e-8,(1-1e-8))

print("----- Training -----")

def get_valid_score(w,b,X_valid,Y_valid):
	z = np.dot(X_valid,np.transpose(w)) +b
	y = sigmoid(z)
	y = np.around(y)
	result = (np.squeeze(Y_valid) == y)
	print("Validation acc = %f" % (float(result.sum())/len(X_valid)) )
	return

percentage = 0.1
X_train, Y_train, X_valid, Y_valid = split_valid_data(data,label,percentage)
X_train, Y_train = _shuffle(X_train,Y_train)

w = np.zeros((106,))
b = np.zeros((1,))
LEARNING_RATE = 0.1
batch_size = 32
step = int(floor(len(X_train))/batch_size)
epoch = 10000

total_loss = 0.0
sum_gradient = 0.0
for epoch in range(epoch):
	total_loss = 0.0
	for i in range(step):
		X = X_train[i*batch_size:(i+1)*batch_size]
		Y = Y_train[i*batch_size:(i+1)*batch_size]
		z = np.dot(X,np.transpose(w)) + b
		y = sigmoid(z) 
		cross_entropy = -1*(np.dot(np.squeeze(Y),np.log(y)) + np.dot(1-np.squeeze(Y),np.log(1-y)))
		total_loss += cross_entropy

		w_grad = np.average(-1 * X *(np.squeeze(Y)-y).reshape((batch_size,1)),axis = 0)
		sum_gradient += w_grad ** 2
		b_grad = np.average(-1 * (np.squeeze(Y)-y) )
		sum_gradient += b_grad **2
		ada = np.sqrt(sum_gradient)

		w = w - LEARNING_RATE * w_grad/sum_gradient
		b = b - LEARNING_RATE * b_grad
	if epoch % 100 == 99:
		print("Iteration: "+str(epoch+1),end=' ')
		get_valid_score(w,b,X_valid,Y_valid)
print("----- Training Result -----")
get_valid_score(w,b,X_valid,Y_valid)
X = testdata
z = np.dot(X,np.transpose(w))+b
y = sigmoid(z)
y = np.around(y)


outputfile = open(argout,'w')
outputfile.write('id,label\n')
for i in range(len(y)):
	outputfile.write(str(i+1)+','+str(int(y[i]))+'\n')
outputfile.close()
print("----- Output Successfully -----")
