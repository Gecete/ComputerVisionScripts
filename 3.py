import cv2, copy
import numpy as np

#We apply parts from exercise 2 to determine the mask by playing with the trackbar values:
# best mask is hsv =(39,191,199) and ranges between low and high of (+-15h, +- 70s, +-70v)

def nothing(x):
	global lower_green, upper_green, mask
	lower_green = np.array([h-15, s-70, v-70])
	upper_green = np.array([h+15, s+70, v+70])
	mask = cv2.inRange(img2, lower_green, upper_green)
	pass

img = cv2.imread('tennis2.jpg')

# Convert BGR to HSV
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgNew = copy.deepcopy(img2)

cv2.namedWindow('mask')
# define range of green color in HSV
lower_green = np.array([39-15,191-70,199-70])
upper_green = np.array([39+15,191+70,199+70])
# Threshold the HSV image to get only green colors

mask = cv2.inRange(img2, lower_green, upper_green)
# Create trackbars for color change
cv2.createTrackbar('H+-15','mask',36,179,nothing)
cv2.createTrackbar('S+-70','mask',191,255,nothing)
cv2.createTrackbar('V+-70','mask',199,255,nothing)

while(1):
	imgOUT=cv2.cvtColor(imgNew, cv2.COLOR_HSV2BGR);
	res = cv2.bitwise_and(img, img, mask=mask)
	cv2.imshow('mask',res )
	cv2.resizeWindow('mask', 300, 300)
	if cv2.waitKey(1) & 0xFF == 27:
		break
	# get current positions of trackbars
	h = cv2.getTrackbarPos('H+-15', 'mask')
	s = cv2.getTrackbarPos('S+-70', 'mask')
	v = cv2.getTrackbarPos('V+-70', 'mask')


cv2.destroyAllWindows()
