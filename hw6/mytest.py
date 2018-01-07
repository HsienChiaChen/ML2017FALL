import numpy as np
import sys
import csv
from math import floor
from random import shuffle


# ========================= #
# Choose the Execution Mode #
# ========================= #

#import time
#st = time.time()

test_csv = sys.argv[1]
dict_file = sys.argv[3]
outputfile_path = sys.argv[2]
# ===================== #
# Loading Training Data #
# ===================== #

print(">>>>> Loading Label Data ...")

labels = []
question = []
dictfile = open(dict_file,'r')
table = csv.reader(dictfile, delimiter = ',')
for r in table:
	labels.append(list(map(int,r[1])))
dictfile.close()

print(">>>>> Loading Test Case Data ...")

rownum = 0
testdata = open(test_csv,'r')
table = csv.reader(testdata, delimiter = ',')
for r in table:
	if rownum !=0:
		l = list(map(int,r[1:]))
		question.append(l)
	rownum+=1
testdata.close()
question = np.array(question)

print(">>>>> Compare 2 Image Lables ...")

ans = []
c = 0
for i in range(len(question)):
	if labels[question[i][0]] == labels[question[i][1]]:
		ans.append(1)
		c+=1
	else: ans.append(0)
print("c =",c)

print(">>>>> Output Predict File ...")
output = open(outputfile_path,'w')
output.write('ID,Ans\n')
for i in range(len(ans)):
	output.write(str(i)+','+str(ans[i])+'\n')
output.close()
print("Output CSV File Successfully !!!")

