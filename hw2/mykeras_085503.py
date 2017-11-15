import numpy as np
import csv
from math import floor
import sys
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
# ------------------------
# Read data from train.csv
# ------------------------

argtrainx = sys.argv[1] # X_train
argtrainy = sys.argv[2] # Y_train
argtest = sys.argv[3] # X_test
argout = sys.argv[4] # outputfile

print("----- Read the data -----")

data = []
rownum = 0
inputfile = open(argtrainx,'r',encoding = 'big5')
table = csv.reader(inputfile, delimiter = ',')
for r in table:
	if rownum !=0:
		data.append([])
		for i in range(len(r)):
			data[rownum-1].append(float(r[i]))
	rownum+=1
inputfile.close()
data = np.array(data)
'''avg = [0.0,0.0,0.0,0.0,0.0]
sd = [0.0,0.0,0.0,0.0,0.0]
count = 0
for i in [0,1,3,4,5]:
	avg[count] = np.average(data[:,i])
	sd[count] = np.average(data[:,i])
	count+=1
count = 0
for i in [0,1,3,4,5]:
	for j in range(len(data)):
		data[j][i] = (data[j][i]-avg[count])/sd[count]
	count+=1'''


label = []
rownum = 0
inputfile = open(argtrainy,'r',encoding = 'big5')
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
inputfile = open(argtest,'r',encoding = 'big5')
table = csv.reader(inputfile, delimiter = ',')
for r in table:
	if rownum != 0:
		testdata.append([])
		for i in range(len(r)):
			testdata[rownum-1].append(float(r[i]))
	rownum+=1
inputfile.close()
testdata = np.array(testdata)
'''avg = [0.0,0.0,0.0,0.0,0.0]
sd = [0.0,0.0,0.0,0.0,0.0]
count = 0
for i in [0,1,3,4,5]:
	avg[count] = np.average(testdata[:,i])
	sd[count] = np.average(testdata[:,i])
	count+=1
count = 0
for i in [0,1,3,4,5]:
	for j in range(len(testdata)):
		testdata[j][i] = (testdata[j][i]-avg[count])/sd[count]
	count+=1'''

#data     = np.array(pd.read_csv('X_train',sep=',',header=0).values)
#label    = np.array(pd.read_csv('Y_train',sep=',',header=0).values)
#testdata = np.array(pd.read_csv('X_test' ,sep=',',header=0).values)


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

percentage = 0.4
X_train, Y_train, X_valid, Y_valid = split_valid_data(data,label,percentage)
X_train, Y_train = _shuffle(X_train,Y_train)


model = Sequential()
model.add(Dense(input_dim = 106,units=20,activation = 'sigmoid'))
for i in range(5):
	model.add(Dense(units = 20,activation = 'sigmoid'))
model.add(Dense(units = 2, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=100,epochs=30)

'''result = model.evaluate(X_valid,Y_valid)
print("loss:",result[0])
print("accuracy:",result[1])'''

result = model.predict(testdata)
#print(result)



res = []
for i in range(len(testdata)):
	if result[i][1]>=0.5:  ans = 1
	else: ans=0
	res.append(ans)

outputfile = open(argout,'w')
outputfile.write('id,label\n')
for i in range(len(res)):
	outputfile.write(str(i+1)+','+str(res[i])+'\n')
outputfile.close()
print("Output Successfully")
