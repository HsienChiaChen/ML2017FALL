import numpy as np
import csv
import math
import sys

argtest = sys.argv[1]
argout = sys.argv[2]

CONTINUE_HOURS = 9
ORDER = 2

w = np.load('model.npy')
CORRECT = False
NORMALIZE = False

# ----------------------
# Out put the prediction
# ----------------------

# 1. read data from test.csv
rel_idx = sorted([9,8,5,6])
test_data = []
test_file = open(argtest,'r',encoding = 'big5')
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

'''elif CORRECT:
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
	'''
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

outputfile = open(argout,'w')
outputfile.write('id,value\n')
for i in range(len(res)):
	outputfile.write(res[i][0]+','+str(res[i][1])+'\n')
outputfile.close()
print("Output Successfully")
