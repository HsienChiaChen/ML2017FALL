import numpy as np
import csv
import math

# ------------------------
# Read data from train.csv
# ------------------------

data = []
for i in range(18):
	data.append([])
rownum = 0
inputfile = open('train.csv','r',encoding = 'big5')
table = csv.reader(inputfile, delimiter = ',')
for r in table:
	if rownum !=0:
		for i in range(3,27):
			if r[i] != 'NR':
				data[(rownum-1)%18].append(float(r[i]))
			else:
				data[(rownum-1)%18].append(0.0)
	rownum+=1
inputfile.close()

CORRECT = False
NORMALIZE = False

if CORRECT:
	for i in range(len(data)):
		for j in range(len(data[i])):
			if data[i][j] < 0:
				k = j-1
				while 1:
					if data[i][k] >= 0:
						data[i][j] = data[i][k]
						break
					else:
						k = k-1

# ------------------------------------------------
# Check the relation between PM2.5 and other index
# ------------------------------------------------

mean = [0.0 for i in range(18)]
std = [0.0 for i in range(18)]
for i in range(18):
	for j in range(480*12):
		mean[i]+=data[i][j]
	mean[i] /= (480*12)
for i in range(18):
	for j in range(480*12):
		std[i] += ((data[i][j]-mean[i])**2)
	std[i]  = math.sqrt(std[i]/(480*12))
#print(mean)
#print(cov)
rel = [0 for i in range(18)]
for i in range(18):
	cov = 0.0
	for j in range(480*12):
		cov += (data[9][j]-mean[9])*(data[i][j]-mean[i])
	cov /= (480*12)
	rel[i] = cov/(std[9]*std[i])
#for i in range(len(rel)):
#	print(str(i)+" "+str(rel[i]))

if NORMALIZE:
	for i in range(len(data)):
		for j in range(len(data[i])):
			data[i][j] = (data[i][j]-mean[i])/std[i]

# --------------------------------------------------
# Create matrix x (data) , y (answer) , w(parameter)
# --------------------------------------------------

rel_rank = [9,8,5,6,12,7,13,3,2,11,1,14,15,16,17,10,4,0]

#rel_idx = [8,9]
#rel_idx = [5,8,9]
#rel_idx = [5,6,7,8,9,12,13]
#rel_idx = [1,2,3,5,6,7,8,9,11,12,13]
#rel_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

# Training Parameter Setting
SAMPLE_PER_MONTH = 100
CONTINUE_HOURS = 9
ORDER = 2
LEARNING_RATE = 10
REPEAT = 5000000
FEATURE_NUM = 4
LAMBDA = 0

rel_idx = sorted(rel_rank[0:FEATURE_NUM])
print(rel_idx)

x = []
y = []
w = []

for i in range(12):
	for j in range(SAMPLE_PER_MONTH):
		x.append([]) # len(x) = 12*471 = 5652
		#for p in range(18): # 18 kinds of index
		for p in rel_idx:
			for q in range(CONTINUE_HOURS):
				b = data[p][480*i+j+q]
				for u in range(ORDER):
					x[SAMPLE_PER_MONTH*i+j].append((b**(u+1)))
		y.append(data[9][480*i+j+CONTINUE_HOURS])
x = np.array(x)
y = np.array(y)
x = np.concatenate((np.ones((x.shape[0],1)),x),axis = 1)

'''

x (5652*163)              y(5652*1)   w(1*163)
[ 1 (9*18 features) ]     [ ans ]     [param]
[ 1       ...       ]     [ ans ]     [param]
[ 1       ...       ]     [ ans ]     [param]

'''

# --------------
# Start training 
# --------------

w = np.zeros(len(x[0]))
sum_gradient = np.zeros(len(x[0]))

for i in range(REPEAT):
	loss = np.dot(x,w) - y
	cost = math.sqrt(np.sum(loss**2) / len(x)) + LAMBDA*np.sum(w**2)
	gradient = np.dot(x.transpose(),loss)
	sum_gradient += sum(gradient ** 2)
	ada = np.sqrt(sum_gradient)
	w = w - LEARNING_RATE*gradient/ada
	if (i+1)%(REPEAT/100)==0:
		print('Iteration: %d , Loss: %f .... %d/100' %(i+1,cost,int((i+1)*100/(REPEAT))))
print(np.sum(w**2))
# -----------------------------------------------
# Self check the corectness of trained parameters
# -----------------------------------------------

print("------Self testing------")
xx = []
yy = []
for i in range(12):
	for j in range(SAMPLE_PER_MONTH+10,471):
		xx.append([])
		for p in rel_idx:
			for q in range(CONTINUE_HOURS):
				b = (461-SAMPLE_PER_MONTH)*i+(j-SAMPLE_PER_MONTH-10)
				z = data[p][480*i+j+q]
				for u in range(ORDER):
					xx[b].append((z**(u+1)))
		yy.append(data[9][480*i+j+CONTINUE_HOURS])
xx = np.array(xx)
yy = np.array(yy)
xx = np.concatenate((np.ones((xx.shape[0],1)),xx),axis=1)

#ans = np.dot(xx,w) * std[9] + mean[9]
#yyy = yy * std[9] + mean[9]

loss = (np.dot(xx,w) - yy)
if NORMALIZE:
	loss = loss * std[9]
#loss = ans-yyy
cost = math.sqrt(np.sum(loss**2)/len(xx))
print("test cost:"+str(cost))


# --------------
# Save Parameter
# --------------

saveparam = open('trainset.csv','w')
for i in range(len(w)):
	saveparam.write(str(w[i]))
	if i != len(w)-1:
		saveparam.write('\n')

# ----------------------
# Out put the prediction
# ----------------------

# 1. read data from test.csv

test_data = []
test_file = open('test.csv','r')
table2 = csv.reader(test_file,delimiter = ',')
rownum=0

if not CORRECT: 
	for r in table2:
		if rownum % 18 == rel_idx[0]:
			test_data.append([])
			for i in range(11-CONTINUE_HOURS,11):
				for u in range(ORDER):
					if NORMALIZE:
						sc = (float(r[i])-mean[rownum%18])/std[rownum%18]
					else:
						sc = float(r[i])
					test_data[rownum//18].append(sc**(u+1))
		elif (rownum % 18) in rel_idx[1:]:
			for i in range(11-CONTINUE_HOURS,11):
				if r[i] != 'NR':
					for u in range(ORDER):
						if NORMALIZE:
							sc = (float(r[i]-mean[rownum%18]))/std[rownum%18]
						else:
							sc = float(r[i])
						test_data[rownum//18].append(float(r[i])**(u+1))
				else:
					for u in range(ORDER):
						if NORMALIZE:
							sc = (0.0-mean[rownum%18])/std[rownum%18]
						else:
							sc = 0.0
						test_data[rownum//18].append(0.0)
		rownum +=1

elif CORRECT:
	for r in table2:
		if rownum % 18 == rel_idx[0]:
			test_data.append([])
			for i in range(11-CONTINUE_HOURS,11):
				for u in range(ORDER):
					if float(r[i]) >= 0.0 :
						if NORMALIZE:
							sc = (float(r[i])-mean[rownum%18])/std[rownum%18]
						else:
							sc = float(r[i])
						test_data[rownum//18].append((sc)**(u+1))
					else:
						k = i-1
						flag = False
						flag2 = True
						if k == 1:
							flag2 = False
							k = i+1
						while flag2:
							if float(r[k]) >= 0.0:
								if NORMALIZE:
									sc = (float(r[k])-mean[rownum%18])/std[rownum%18]
								else:
									sc = float(r[k])
								test_data[rownum//18].append((sc)**(u+1))
								flag = True
								break
							else:
								if k == 1:
									k = i+1
									break
								else:
									k = k-1
						flag2 = True
						if k == 11:
							flag2 = False
							if NORMALIZE:
								sc = (0.0-mean[rownum%18])/std[rownum%18]
							else:
								sc = 0.0
							test_data[rownum//18].append(sc**(u+1))
						while (not flag) and flag2:
							if float(r[k]) >= 0.0:
								if NORMALIZE:
									sc = (float(r[k])-mean[rownum%18])/std[rownum%18]
								else:
									sc = float(r[k])
								test_data[rownum//18].append((sc)**(u+1))
								break
							else:
								if k == 10:
									test_data[rownum//18].append(0.0)
									break
								else:
									k = k+1
								
		elif (rownum % 18) in rel_idx[1:]:
			for i in range(11-CONTINUE_HOURS,11):
				if r[i] != 'NR':
					for u in range(ORDER):
						if float(r[i]) >= 0.0:
							if NORMALIZE:
								sc = (float(r[i])-mean[rownum%18])/std[rownum%18]
							else:
								sc = float(r[i])
							test_data[rownum//18].append((sc)**(u+1))
						else:
							k = i-1
							flag = False
							flag2 = True
							if k == 1:
								flag2 = False
								k = i+1
							while flag2:
								if float(r[k]) >= 0.0:
									if NORMALIZE:
										sc = (float(r[k])-mean[rownum%18])/std[rownum%18]
									else:
										sc = float(r[k])
									test_data[rownum//18].append((sc)**(u+1))
									flag = True
									break
								else:
									if k == 1:
										k = i+1
										break
									else:
										k = k-1
							flag2 = True
							if k == 11:
								flag2 = False
								if NORMALIZE:
									sc = (0.0-mean[rownum%18])/std[rownum%18]
								else:
									sc = 0.0
								test_data[rownum//18].append(sc**(u+1))
							while (not flag) and flag2:
								if float(r[k]) >= 0.0:
									if NORMALIZE:
										sc = (float(r[k])-mean[rownum%18])/std[rownum%18]
									else:
										sc = float(r[k])
									test_data[rownum//18].append((sc)**(u+1))
									break
								else:
									if k == 10:
										test_data[rownum//18].append(0.0)
										break
									else:
										k = k+1
				else:
					for u in range(ORDER):
						if NORMALIZE:
							sc = (0.0-mean[rownum%18])/std[rownum%18]
						else:
							sc = 0.0
						test_data[rownum//18].append(sc)
			
		rownum += 1
	
test_file.close()

test_data = np.array(test_data)
test_data = np.concatenate((np.ones((test_data.shape[0],1)),test_data),axis=1)

# 2. Calculate the predict PM2.5

res = []
for i in range(len(test_data)):
	res.append(['id_'+str(i)])
	if NORMALIZE:
		ans = np.dot(w,test_data[i])*std[9]+mean[9]
	else:
		ans = np.dot(w,test_data[i])
	res[i].append(ans)

outputfile = open('result.csv','w')
outputfile.write('id,value\n')
for i in range(len(res)):
	outputfile.write(res[i][0]+','+str(res[i][1])+'\n')
outputfile.close()
