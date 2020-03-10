#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:59:30 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp 
import scipy.ndimage as ndimage


img = cv2.imread("original_retinal_images/IDRiD_01.jpg",0)
#img = cv2.imread("img2.jpg",1)


#img_gray = cv2.imread("original_retinal_images/IDRiD_05.jpg",0)

ret,thresh1 = cv2.threshold(img,170,255,cv2.THRESH_BINARY)


cv2.imwrite("temp.jpg",thresh1)




#plt.hist(img_gray.ravel(),256,[0,256]); plt.show()

#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv2.calcHist([img],[i],None,[256],[0,256])
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])
#plt.show()