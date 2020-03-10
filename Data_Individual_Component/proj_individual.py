import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp 
import scipy.ndimage as ndimage

img1 = cv2.imread("original_retinal_images/IDRiD_49.jpg",1)
gt1 = cv2.imread("optic_disc_segmentation_masks/IDRiD_08_OD.tif",0)


def preprocess(img):
    b,g,r = cv2.split(img)
    gray = rgb2Red(img)
    gray_blur = cv2.GaussianBlur(g, (5,5), 5)
#    gray = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0, gray)
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
#    gray = ndimage.grey_closing(gray,structure=kernel)
#    gray = cv2.equalizeHist(gray)
    return gray_blur

def ROI(img):
    b,g,r = cv2.split(img)
    g = cv2.GaussianBlur(g,(5,5),0)
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
#    g = ndimage.grey_opening(g,structure=kernel)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)
    x0 = int(maxLoc[0])-510
    y0 = int(maxLoc[1])-510
    x1 = int(maxLoc[0])+510
    y1 = int(maxLoc[1])+510
    return g[y0:y1,x0:x1]


temp = img1.copy()

def getROI(image):
	image_resized = image.copy()
	b,g,r = cv2.split(image_resized)
	g = cv2.GaussianBlur(g,(15,15),0)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
	g = ndimage.grey_opening(g,structure=kernel)	
	(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)

	x0 = int(maxLoc[0])-510
	y0 = int(maxLoc[1])-510
	x1 = int(maxLoc[0])+510
	y1 = int(maxLoc[1])+510
	
	return image_resized[y0:y1,x0:x1]
#blurred = cv2.GaussianBlur(temp, (91, 91), 0)
#


#ret,thresh1 = cv2.threshold(temp,165,255,cv2.THRESH_BINARY)


#circles = cv2.HoughCircles(thresh1, cv2.HOUGH_GRADIENT, 1.2, 500)

#thresh1 = preprocess(temp)
#cv2.imwrite("img49.jpg",thresh1)

#ii = cv2.imread("img49.jpg",0)
#ret,thresh1 = cv2.threshold(thresh1,236,255,cv2.THRESH_BINARY)
#thresh1 = cv2.Canny(thresh1,240,250)
thresh1 = getROI(img1)

plt.imshow(thresh1)




