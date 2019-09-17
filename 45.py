import cv2
import numpy as np
#Five tennis ball images, tennis1.jpg, tennis2.jpg, tennis3.jpg...
for i in range(1,7):
	img = cv2.imread('tennis'+str(i)+'.jpg')

	# Convert BGR to HSV
	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	#from previous exercise I chose the best values for the mask (35,185,186)
	(h,s,v)=(35,185,186)
	lower_green = np.array([h-7, s-70, v-70])
	upper_green = np.array([h+7, s+70, v+70])
	mask = cv2.inRange(img2, lower_green, upper_green)

	kernel = np.ones((5,5),np.uint8)
	#cv2.imshow('mask', mask)
	opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,iterations = 2)
	opening = cv2.medianBlur(mask, 7)

	opening = cv2.erode(opening,kernel,iterations = 1)
	cv2.imshow('mask'+str(i), opening)
	cv2.imwrite('mask'+str(i)+'.jpg', opening)
	cv2.moveWindow('mask' + str(i), 220 * (i - 1), 50 * i+150)
	imgray = cv2.cvtColor(opening,cv2.COLOR_BAYER_RG2GRAY)
	ret,thresh = cv2.threshold(imgray,127,255,0)
	im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(img, contours, -1, (0,0,255), 2)
	cont=0
	for cnt in contours:
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		center = (int(x),int(y))
		radius = int(radius)
		if radius>2:
			cont=cont+1
			cv2.circle(img,center,radius+3,(0,0,255),2)
	if cont > 1:
		message=' Tennis balls'
	else:
		message=' Tennis ball'
	cv2.putText(img,str(cont)+message ,(5, 20), cv2.FONT_ITALIC, 0.4, (0, 216, 255), 1, cv2.LINE_AA)

	cv2.imshow('image' + str(i), img);
	cv2.imwrite('detect' + str(i)+'.jpg', img)
	cv2.moveWindow('image' + str(i), 220 * (i-1), 40 * i)
cv2.waitKey()