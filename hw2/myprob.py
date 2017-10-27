import numpy as np
import csv
from math import log,floor
from random import shuffle
import sys
import pandas as pd
# ------------------------
# Read data from train.csv
# ------------------------


argtrainx = sys.argv[1] # X_train
argtrainy = sys.argv[2] # Y_train
argtest = sys.argv[3] # X_test
argout = sys.argv[4] # outputfile

#np.random.seed(2401)
print("----- Read the data -----")

data     = np.array(pd.read_csv(argtrainx,sep=',',header=0).values)
label    = np.array(pd.read_csv(argtrainy,sep=',',header=0).values)
testdata = np.array(pd.read_csv(argtest  ,sep=',',header=0).values)

def _shuffle(X,Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize],Y[randomize])

print(data.shape)
print(label.shape)
print(testdata.shape)

print("----- Normalize data -----")

alldata = np.concatenate((data,testdata))
mean = np.average(alldata,axis = 0)
sigma = np.std(alldata,axis = 0)
mean = np.tile(mean,(alldata.shape[0],1))
sigma = np.tile(sigma,(alldata.shape[0],1))
normed_data = (alldata-mean) / sigma

data = normed_data[:data.shape[0]]
testdata = normed_data[data.shape[0]:]

def split_valid_data(data,label,percentage):
	N_train = int(floor(data.shape[0]*percentage))
	data,label = _shuffle(data,label)
	X_valid = data[0:N_train]
	Y_valid = label[0:N_train]
	X_train = data[N_train:]
	Y_train = label[N_train:]
	return X_train, Y_train, X_valid, Y_valid
	
def sigmoid(z):
	res = 1/ (1.0+np.exp(-z))
	return np.clip(res,1e-8,(1-1e-8))

percentage = 0.1
X_train, Y_train, X_valid, Y_valid = split_valid_data(data,label,percentage)
#X_train, Y_train = _shuffle(X_train,Y_train)


print("----- Split the data into 2 subdata -----")
data0 = [] # Class 0
data1 = [] # Class 1
for i in range(len(X_train)):
	if Y_train[i] == 1:
		data0.append(X_train[i])
	else:
		data1.append(X_train[i])

data0 = np.array(data0)
data1 = np.array(data1)
print(data0.shape)
print(data1.shape)

print("----- Calculate Mean and Covariance -----")
avg0 = np.zeros((106,))
avg1 = np.zeros((106,))
for i in range(len(data0)):
	avg0 += data0[i]
for i in range(len(data1)):
	avg1 += data1[i]

avg0 /= data0.shape[0]
avg1 /= data1.shape[0]


cov0 = np.zeros((106,106))
cov1 = np.zeros((106,106))

N0 = data0.shape[0]
N1 = data1.shape[0]

for i in range(data0.shape[0]):
	cov0 += np.dot(np.transpose([data0[i]-avg0]),[data0[i]-avg0])
for i in range(data1.shape[0]):
	cov1 += np.dot(np.transpose([data1[i]-avg1]),[data1[i]-avg1])


cov0 /= N0
cov1 /= N1


def sigmoid(t):
	return 1 / (1+np.exp(-t))

cov = (float(N0)*cov0 + float(N1)*cov1)/float(N0+N1)
inverse = np.linalg.inv(cov)
print('---compute w---')
w = np.dot((avg0 - avg1),inverse) # 1*106

b = (-0.5)*np.dot(np.dot([avg0],inverse),avg0)+0.5*np.dot(np.dot([avg1],inverse),avg1)+np.log(float(N0)/N1)


x = X_valid.T
a = sigmoid(np.dot(w,x) + b )
y = np.around(a)
result = ( np.squeeze(Y_valid) == y)

print("Valid acc = %f" % (float(result.sum())/len(result) )    )
print("Predicting")
x = testdata.T
res = sigmoid( np.dot(w,x) + b )
res = np.around(res)

outputfile = open(argout,'w')
outputfile.write('id,label\n')
for i in range(len(res)):
	outputfile.write(str(i+1)+','+str(int(res[i]))+'\n')
outputfile.close()
print("Output Successfully")
