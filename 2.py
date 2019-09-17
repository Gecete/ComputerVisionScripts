import cv2, copy
import numpy as np

def nothing(x):
	global imgNew, flag
	print(h,s,v)
	flag=1
	print(img2[0,0,2])
	if imgNew.shape[0]>img2.shape[0]:
		imgNew = imgNew[30:imgNew.shape[0],:imgNew.shape[1]]

	imgNew [:,:,2] = img2[:,:,2]*float(v)/255;
	imgNew [:,:,1] = img2[:,:,1]*float(s)/255;
	imgNew [:,:,0] = float(h);

	pass
def showHSVvalues(event,x,y,flags,param):
	global imgNew, flag

	if event == cv2.EVENT_LBUTTONDBLCLK:
		if flag == 1:
			imgNew = img2
			flag = 0
		w=imgNew.shape[1]
		borderTop=np.zeros((30,w,3), np.uint8)

		hueValue = int(imgNew[y, x, 0])
		satValue = int(imgNew[y, x, 1])
		valValue = int(imgNew[y, x, 2])
		if imgNew.shape[0]==img2.shape[0]:

			imgNew = np.concatenate((borderTop, imgNew), axis=0)
		else:
			imgNew[:int(borderTop.shape[0]),:int(borderTop.shape[1])]=borderTop

		cv2.circle(imgNew, (imgNew.shape[1]-30, imgNew.shape[0]-30), 30, (hueValue, satValue, valValue), -1)
		cv2.putText(imgNew, "HSV value in click (x,y): (" + str(hueValue)+" , "+str(satValue)+" , "+str(valValue)+")",
					(5, 20), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

flag=1
img = cv2.imread('tennis2.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgNew = copy.deepcopy(img2)
hueValue=''
cv2.namedWindow('image')

cv2.setMouseCallback('image',showHSVvalues)

# Create trackbars for color change
cv2.createTrackbar('H','image',0,179,nothing)
cv2.createTrackbar('S','image',255,255,nothing)
cv2.createTrackbar('V','image',255,255,nothing)
cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)

while(1):
	imgOUT=cv2.cvtColor(imgNew, cv2.COLOR_HSV2BGR);

	cv2.imshow('image',imgOUT);
	cv2.resizeWindow('image', 300, 300)
	if cv2.waitKey(1) & 0xFF == 27:
		break
	# get current positions of trackbars
	h = cv2.getTrackbarPos('H','image')
	s = cv2.getTrackbarPos('S','image')
	v = cv2.getTrackbarPos('V','image')


cv2.destroyAllWindows()
