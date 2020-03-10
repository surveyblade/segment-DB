#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:46:37 2019

@author: user
"""

import numpy as np
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
import cv2


def resize(img):
	width = 1024
	height = 720
	return cv2.resize(img,(width,height), interpolation = cv2.INTER_CUBIC)

img = cv2.imread("original_retinal_images/IDRiD_11.jpg",1)

#get roi
image_resized = resize(img)
b,g,r = cv2.split(image_resized)
g = cv2.GaussianBlur(g,(15,15),0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
g = ndimage.grey_opening(g,structure=kernel)	
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)
x0 = int(maxLoc[0])-110
y0 = int(maxLoc[1])-110
x1 = int(maxLoc[0])+110
y1 = int(maxLoc[1])+110
roi = g[y0:y1,x0:x1]

resized = resize(img)
ret,thresh1 = cv2.threshold(resized,160,255,cv2.THRESH_BINARY)
gray = cv2.cvtColor(thresh1,cv2.COLOR_BGR2GRAY)

#ret,gray = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
row,col = gray.shape


for row in range(len(gray)):
    for col in range(len(gray[row])):
        if row > y0 and row < y1:
            if col > x0 and col < x1:
                continue
            else:
                gray[row][col] = 0
        else:
            gray[row][col] = 0

plt.imshow(gray)

cv2.imwrite("result.jpg",gray)