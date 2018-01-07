import numpy as np
import skimage.io
import sys
from numpy.linalg import svd,eig
import os

#import matplotlib.pyplot as plt

print("Loading image ...")
dir_path = sys.argv[1]
img_path = sys.argv[2]

data = []
r_img = np.array(skimage.io.imread(os.path.join(dir_path,img_path)),np.float32)
for i in range(415):
	file_name = os.path.join(dir_path,(str(i) + ".jpg"))
	data.append(np.array(skimage.io.imread(file_name),np.float32).flatten())

avg_face = np.mean(data,axis = 0).reshape(600,600,3)

print(avg_face.shape)
#plt.imshow(np.array(avg_face,np.uint8))
#plt.show()
#skimage.io.imsave("avg_face.jpg",np.array(avg_face,np.uint8))


print("Doing SVD ...")
X = data - avg_face.flatten()
#print(X.shape)
u,s,v = svd(np.transpose(X),full_matrices = False)
#print("u",u.shape)
#print("s",s.shape)
#print("v",v.shape)
'''for i in range(4):
	#plt.subplot(2,5,i+1)
	M = np.transpose(u[:,i])*(-1)
	M -= np.min(M)
	M /= np.max(M)
	M = (M*255).astype(np.uint8)
	print(M)
	#plt.imshow(M.reshape(600,600,3))
	#plt.xticks([])
	#plt.yticks([])
	e_name = "eigenface_"+str(i+1)+".jpg"
	skimage.io.imsave(e_name,M.reshape(600,600,3))
#plt.show()'''


print("Reconstruct the specific image")
y = np.transpose(r_img.flatten())
#print("y",y.shape)
#print("u[:,:4]",u[:,:4].shape)
yu = np.dot(y,u[:,:4])
#print("yu",yu.shape)
sums = np.sum(s)
#for i in range(10):
#	print("lambda/sum =",i,":",float(s[i])/float(sums))

restruct = np.dot(yu,np.transpose(u[:,:4]))
print("restruct",restruct.shape)
#plt.subplot(1,2,1)
#plt.imshow(y.reshape(600,600,3).astype(np.uint8))
#plt.subplot(1,2,2)
#plt.imshow(restruct.reshape(600,600,3).astype(np.uint8))
#plt.show()
skimage.io.imsave("reconstruction.jpg",(restruct.reshape(600,600,3)+avg_face).astype(np.uint8))

'''print("Pick 4 image to reconstruct with 4 eigenfaces ...")
idx = [100,200,300,400]
for i in idx:
	y = np.transpose(data[i])
	yu = np.dot(y,u[:,:4])
	restruct = np.dot(yu,np.transpose(u[:,:4]))
	f_name = "reconstruction_" + str(i) + ".jpg"
	skimage.io.imsave(f_name,(restruct.reshape(600,600,3)+avg_face).astype(np.uint8))'''
#plt.show()
print("Done !!!")
