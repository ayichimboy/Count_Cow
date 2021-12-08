#import
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils

#read image
drone = cv.imread('cows.jpg')
#drone = cv.imread('herd_2.jpg')
cv.imshow('FPV', drone)

#convert to grayscale
drone_gray = cv.cvtColor(drone, cv.COLOR_BGR2GRAY)
#cv.imshow('FPV-GRAY', drone_gray)

#blurr
blur = cv.GaussianBlur(drone_gray, (15, 15), cv.BORDER_DEFAULT)
#cv.imshow('blur', blur)

#global threshold
glob, thresh = cv.threshold(drone_gray, 127, 255, cv.THRESH_BINARY)
#cv.imshow('global threshold', thresh)

glob1, thresh1 = cv.threshold(blur,1 ,255, cv.THRESH_BINARY+cv.THRESH_OTSU)
#cv.imshow('otsu', thresh1)

#adaptive thresholding
cow_cnt = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 255, 19)
cv.imshow('MEAN', cow_cnt)

cow_cnt_1 = cv.adaptiveThreshold(blur, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 255, 19)
cv.imshow('GAUSSIAN', cow_cnt_1)

cow_cnt_1=cv.bitwise_not(cow_cnt_1)
cv.imshow('BITWISE', cow_cnt_1)

#Erosion and Dilatation
# kernel=np.ones((15,15), np.uint8)
# cow_dil = cv.dilate(cow_cnt, kernel, iterations=1)
# cow_erode = cv.erode(cow_dil, kernel, iterations=1)
# cv.imshow('ERODE', cow_erode)

#Label image
ret, labels = cv.connectedComponents(cow_cnt_1)
label_hue = np.uint8(179*labels/np.max(labels))
black_ch = 225 * np.ones_like(label_hue)
label_cow = cv.merge([label_hue, black_ch, black_ch])

label_cow = cv.cvtColor(label_cow, cv.COLOR_HSV2BGR)
label_cow[label_hue==0]=0
cv.imshow('LABEL', label_cow)

#cv.drawContours()
# cnts = cv.findContours(cow_cnt_1.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
#
# #cv.drawContours(org, cnts, -7, (240, 1, 159), 4)
# cv.drawContours(cow_cnt_1, cnts, -3, (240, 1, 159), 3)
#
# cv.imshow("CONTOUR", cnts)
# cv.imshow("CONTOUR", drone)


#plt.subplot(222)
#plt.title('objects counted:' + str(ret-1))
plt.title('objects counted:' + str(ret))
plt.imshow(label_cow)
#print('Cows Counted:', ret-1)
print('Cows Counted:', ret)
plt.show()




cv.waitKey(0)