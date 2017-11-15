import numpy as np
import csv
from math import log,floor
<<<<<<< HEAD
=======
from random import shuffle
>>>>>>> 25fd36f6e913df9731fe51492a8b068db6ba137f
import sys
import pandas as pd
# ------------------------
# Read data from train.csv
# ------------------------


argtrainx = sys.argv[1] # X_train
argtrainy = sys.argv[2] # Y_train
argtest = sys.argv[3] # X_test
argout = sys.argv[4] # outputfile

<<<<<<< HEAD

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
data = np.array(data)
for i in range(len(data)):
	data[i][0] /= 100
	data[i][1] /= 1000000
	data[i][3] /= 10000
	data[i][4] /= 10000
	data[i][5] /= 100


label = []
rownum = 0
inputfile = open('Y_train','r',encoding = 'big5')
table = csv.reader(inputfile, delimiter = ',')
for r in table:
	if rownum!=0:
		label.append([])
		for i in range(len(r)):
			label[rownum-1].append(1-int(r[i]))
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
inputfile.close()
testdata = np.array(testdata)

for i in range(len(testdata)):
	testdata[i][0] /= 100
	testdata[i][1] /= 1000000
	testdata[i][3] /= 10000
	testdata[i][4] /= 10000
	testdata[i][5] /= 100'''

=======
#np.random.seed(2401)
print("----- Read the data -----")

>>>>>>> 25fd36f6e913df9731fe51492a8b068db6ba137f
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
<<<<<<< HEAD
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
=======
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
>>>>>>> 25fd36f6e913df9731fe51492a8b068db6ba137f
	return X_train, Y_train, X_valid, Y_valid
	
def sigmoid(z):
	res = 1/ (1.0+np.exp(-z))
	return np.clip(res,1e-8,(1-1e-8))

percentage = 0.1
X_train, Y_train, X_valid, Y_valid = split_valid_data(data,label,percentage)
<<<<<<< HEAD
X_train, Y_train = _shuffle(X_train,Y_train)
=======
#X_train, Y_train = _shuffle(X_train,Y_train)

>>>>>>> 25fd36f6e913df9731fe51492a8b068db6ba137f

print("----- Split the data into 2 subdata -----")
data0 = [] # Class 0
data1 = [] # Class 1
for i in range(len(X_train)):
	if Y_train[i] == 1:
<<<<<<< HEAD
		data0.append(data[i])
	else:
		data1.append(data[i])
=======
		data0.append(X_train[i])
	else:
		data1.append(X_train[i])
>>>>>>> 25fd36f6e913df9731fe51492a8b068db6ba137f

data0 = np.array(data0)
data1 = np.array(data1)
print(data0.shape)
print(data1.shape)

<<<<<<< HEAD

print("----- Calculate Mean and Covariance -----")

#sumx = np.sum(data,axis = 0)
#print("sum")
#print(sumx)
avg0 = np.average(data0,axis = 0)
avg1 = np.average(data1,axis = 0)
=======
print("----- Calculate Mean and Covariance -----")
avg0 = np.zeros((106,))
avg1 = np.zeros((106,))
for i in range(len(data0)):
	avg0 += data0[i]
for i in range(len(data1)):
	avg1 += data1[i]

avg0 /= data0.shape[0]
avg1 /= data1.shape[0]

>>>>>>> 25fd36f6e913df9731fe51492a8b068db6ba137f

cov0 = np.zeros((106,106))
cov1 = np.zeros((106,106))

<<<<<<< HEAD
N0 = len(data0)
N1 = len(data1)

'''for i in range(106):
	for j in range(i):
		x = (data0.transpose())[i] - avg0[i]  # item i 1*N
		y = (data0.transpose())[j] - avg0[j]  # item j 1*N
		cov0[i][j] = np.dot(x,y.transpose())
		cov0[j][i] = cov0[i][j]	
		x = (data1.transpose())[i] - avg1[i]  # item i 1*N
		y = (data1.transpose())[j] - avg1[j]  # item j 1*N
		cov1[i][j] = np.dot(x,y.transpose())
		cov1[j][i] = cov1[i][j]
for i in range(106):
	x = (data0.transpose())[i] - avg0[i] 
	cov0[i][i] = np.dot(x,x.transpose())
	x = (data1.transpose())[i] - avg1[i]
	cov1[i][i] = np.dot(x,x.transpose())'''

for i in range(len(avg0)):
	cov0 += np.dot((data0-avg0).transpose(),(data0-avg0))
for i in range(len(avg1)):
	cov1 += np.dot((data1-avg1).transpose(),(data1-avg1))


avg0 = np.array(avg0)
avg1 = np.array(avg1)
cov0 = np.array(cov0)/N0
cov1 = np.array(cov1)/N1
=======
N0 = data0.shape[0]
N1 = data1.shape[0]

for i in range(data0.shape[0]):
	cov0 += np.dot(np.transpose([data0[i]-avg0]),[data0[i]-avg0])
for i in range(data1.shape[0]):
	cov1 += np.dot(np.transpose([data1[i]-avg1]),[data1[i]-avg1])


cov0 /= N0
cov1 /= N1
>>>>>>> 25fd36f6e913df9731fe51492a8b068db6ba137f


def sigmoid(t):
	return 1 / (1+np.exp(-t))

<<<<<<< HEAD
cov = (N0*cov0 + N1*cov1)/(N0+N1)
inverse = np.linalg.pinv(cov)
print(inverse)
w = np.dot((avg0 - avg1).transpose(),inverse) # 1*106
b11 = -0.5*np.dot(avg0.transpose(),inverse) # 1*106
b12 = np.dot(b11,avg0) # scalar
b21 = 0.5*np.dot(avg1.transpose(),inverse) # 1*106
b22 = np.dot(b21,avg1) # scalar
b3 = np.log(N0/N1) # scalar
b = b12 + b22 + b3


y = sigmoid(np.dot(w,X_valid.transpose()) + b )
y = np.around(y)
result = ( np.squeeze(Y_valid) == y)

print("Valid acc = %f" % (float(result.sum())/len(result) )    )

res = sigmoid( np.dot(w,testdata.transpose()) + b )
=======
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
>>>>>>> 25fd36f6e913df9731fe51492a8b068db6ba137f
res = np.around(res)

outputfile = open(argout,'w')
outputfile.write('id,label\n')
for i in range(len(res)):
	outputfile.write(str(i+1)+','+str(int(res[i]))+'\n')
outputfile.close()
print("Output Successfully")
