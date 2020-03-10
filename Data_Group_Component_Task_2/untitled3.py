#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:42:47 2019

@author: user
"""


import numpy as np
import cv2

img = cv2.imread("Test/original_retinal_images/01_test.tif",1)

print(img.shape)