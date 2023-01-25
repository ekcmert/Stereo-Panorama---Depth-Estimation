import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import time



mainfolder = 'images'
myfolder = os.listdir(mainfolder)
print(myfolder)
left_image = cv2.imread('images/hl.PNG',cv2.IMREAD_GRAYSCALE)
#left_image = cv2.blur(left_image,(5,5))
right_image = cv2.imread('images/hr.PNG',cv2.IMREAD_GRAYSCALE)
#right_image = cv2.blur(right_image,(5,5))
#left_image = cv2.resize(left_image,(1600,1600),None,1,1)
#right_image = cv2.resize(right_image,(1600,1600),None,1,1)
stereo = cv2.StereoBM_create(numDisparities=16,blockSize=7)
dept = stereo.compute(left_image,right_image)
#cv2.imshow('left',left_image)
#cv2.imshow('right', right_image)
#cv2.imshow('dept',dept)
cv2.waitKey(0)
plt.imshow(dept)
#plt.axis('on')
plt.show()
#left_image1 = cv2.imread('images/revl7.jpeg',cv2.IMREAD_GRAYSCALE)
#left_image1 = cv2.blur(left_image,(5,5))
#right_image1 = cv2.imread('images/revr7-1.jpeg',cv2.IMREAD_GRAYSCALE)
#right_image1 = cv2.blur(right_image,(5,5))
#left_image1 = cv2.resize(left_image,(1600,1600),None,1,1)
#right_image1 = cv2.resize(right_image,(1600,1600),None,1,1)
#stereo1 = cv2.StereoBM_create(numDisparities=32,blockSize=5)
#dept1 = stereo.compute(left_image,right_image)
#dept1= cv2.rotate(dept1,90)
#cv2.imshow('left',left_image)
#cv2.imshow('right', right_image)
#cv2.imshow('dept1',dept1)
cv2.waitKey(0)
#plt.imshow(dept1)
#plt.axis('on')
plt.show()
#dept1 = cv2.rotate(dept1, cv2.ROTATE_90_COUNTERCLOCKWISE)
#togeather = cv2.add(dept,dept1)
#cv2.imshow(togeather)
#plt.imshow(togeather)

cv2.waitKey(10)