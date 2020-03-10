#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:30:39 2019

@author: user
"""
import numpy as np
import gif2numpy
import math
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


num = 20

#for i in range(1,num+1):
#    if i < 10:
#        gt = "blood_vessel_segmentation_masks/0" + str(i) +"_manual1.gif"
#    else:
#        gt = "blood_vessel_segmentation_masks/" + str(i) +"_manual1.gif"
#    np_frames, extensions, image_specifications = gif2numpy.convert(gt)
#    if i < 10:
#        cv2.imwrite("gt/gt0"+str(i)+".png",np_frames[0])
#    else:
#        cv2.imwrite("gt/gt"+str(i)+".png",np_frames[0])

gt = cv2.imread("gt/gt01.png",0)
gray = cv2.imread("gray/gray01.png",0)

def jaccard_ac(pre,img):
    num_img = cv2.countNonZero(img)
    num_pre = cv2.countNonZero(pre)
    cop = img.copy()
    cop[pre==0] = 0
    joint = cv2.countNonZero(cop)
    jac = joint/(num_img+num_pre-joint)
    return jac

# true Positive / (True Positive + False Negative)
def recall(pre,img):
    num_pre = cv2.countNonZero(img)
    cop = img.copy()
    cop[pre==0]=0
    tp = cv2.countNonZero(cop)
    return tp/num_pre

def accuracy(pre,img):
    _,img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    _,pre = cv2.threshold(pre,1,255,cv2.THRESH_BINARY)
    return accuracy_score(img,pre)

if __name__ == '__main__':
    pre = cv2.imread('5.jpg',0)
    img = cv2.imread('masks_Hard_Exudates/IDRiD_55_EX.tif',0)


#def recall(gt,gray):
#    rows,cols = gt.shape
#    gray_nonzero = cv2.countNonZero(gray)
#    gt_nonzero = cv2.countNonZero(gt)
#    gt[gray==0]=0
#    count = cv2.countNonZero(gt)
##    jaccardCOE = count/(gt_nonzero+gray_nonzero-count)
#    recall = count/gt_nonzero
#    return recall



rec = []
for i in range(1,num+1):
    if i < 10:
        gt = "gt/gt0"+str(i)+".png"
        gray = "gray/gray0" + str(i) + ".png"
    else:
        gt = "gt/gt"+str(i)+".png"
        gray = "gray/gray" + str(i) + ".png"
    gt = cv2.imread(gt,0)
    gray = cv2.imread(gray,0)
    rec.append(jaccard_ac(gray,gt))

print(rec)



        
        
        
        
        
        