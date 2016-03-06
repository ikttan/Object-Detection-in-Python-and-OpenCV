# (c) Copyright 2016, Ian K T Tan
# Multimedia University

import numpy as np
import cv2
import time

# Load an color image in grayscale
# Show the loaded image
img = cv2.imread('football.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Do a threshold on the binary of the image
# Show the resulting image
img_th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,29,69)
cv2.imshow('image',img_th)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect the contours
# Show the resulting image
contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print len(contours)
cnt = contours[0]
print len(cnt)
time.sleep(5.5)

for h,cnt in enumerate(contours):
    mask = np.zeros(img.shape,np.uint8)
    if len(cnt) > 20 :
        cv2.drawContours(mask,[cnt],0,(255,0,0),-1)
        mean = cv2.mean(img,mask = mask)
        print cnt
        time.sleep(3)

