import cv2
import numpy as np

# Define mouse callback function
def draw_circle(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDBLCLK:
  cv2.circle(img,(x,y),50,(0,255,0),-1)
  cv2.imshow('image', img)
# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
# Show image
cv2.imshow('image',img)
cv2.waitKey()
cv2.destroyAllWindows()