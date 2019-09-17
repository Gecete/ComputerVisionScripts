import cv2
import numpy as np

##############################################################
### FUNCTIONS
##############################################################

### 1. Extract ORB keypoints and descriptors from a gray image
def extract_features(gray):

  ## TODO: Detect ORB features and compute descriptors.
  ## TODO: (Overwrite the following 2 lines with your answer.)
  descriptors = np.array([[1,1], [7,5], [5,2], [3,4]], np.float32)
  keypoints = [cv2.KeyPoint(100 * x, 100 * y, 1) for (x,y) in descriptors]

  return (keypoints, descriptors)


### 2. Find corresponding features between the images
def find_matches(keypoints1, descriptors1, keypoints2, descriptors2):

  ## TODO: Look up corresponding keypoints.
  ## TODO: (Overwrite the following 2 lines with your answer.)
  points1 = np.array([k.pt for k in keypoints1], np.float32)
  points2 = np.array([k.pt for k in keypoints2], np.float32)

  return (points1, points2)


### 3. Find homography between the points
def find_homography(points1, points2):

    # convert the keypoints from KeyPoint objects to NumPy arrays
    src_pts = points2.reshape(-1,1,2)
    dst_pts = points1.reshape(-1,1,2)

    # find homography
    homography, mask = cv2.findHomography(src_pts, dst_pts)

    return (homography)


### 4.1 Calculate the size and offset of the stitched panorama
def calculate_size(image1, image2, homography):

    # compute de coordinates of image2 corners in image1
    h2 = img2.shape[0]
    w2 = img2.shape[1]
    corners2 = np.float32([[[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]])
    transformedCorners2 = cv2.perspectiveTransform(corners2, H)

    ## TODO: Calculate the size and offset of the stitched panorama.
    ## TODO: (Overwrite the following 2 lines with your answer.)
    offset = (0, 0)
    size = (1600, 600)

    ## Update the homography to shift by the offset
    homography[0:2, 2] += offset

    return (size, offset)


## 4.2 Combine images into a panorama
def merge_images(image1, image2, homography, size, offset):

  ## TODO: Combine the two images into one.
  ## TODO: (Overwrite the following 5 lines with your answer.)
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  panorama[:h1, :w1] = image1
  panorama[:h2, w1:w1+w2] = image2

  return panorama


### --- No need to change anything below this point ---

### Connects corresponding features in the two images using yellow lines
def draw_matches(image1, image2, points1, points2):

  # Put images side-by-side into 'image'
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  image = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
  image[:h1, :w1] = image1
  image[:h2, w1:w1 + w2] = image2

  # Draw yellow lines connecting corresponding features.
  for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
    cv2.line(image, (x1, y1), (x2 + w1, y2), (0, 255, 255))

  return image


##############################################################
### MAIN PROGRAM
##############################################################

### Load images
img1 = cv2.imread('Image1.jpg')
img2 = cv2.imread('Image2.jpg')

# Convert images to grayscale (for ORB detector).
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

### 1. Detect features and compute descriptors.

(kp1, desc1) = extract_features(img1)
(kp2, desc2) = extract_features(img2)
print ('{0} features detected in image1').format(len(kp1))
print ('{0} features detected in image2').format(len(kp2))

orb1 = cv2.drawKeypoints(gray1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
orb2 = cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Image1_orf.JPG', orb1)
cv2.imwrite('Image2_orb.JPG', orb2)
cv2.imshow('Features 1', orb1)
cv2.imshow('Features 2', orb2)
cv2.waitKey(0)

### 2. Find corresponding features

(points1, points2) = find_matches(kp1, desc1, kp2, desc2)
print ('{0} features matched').format(len(points1))

match = draw_matches(img1, img2, points1, points2)
cv2.imwrite('matching.JPG', match)
cv2.imshow('Matching', match)
cv2.waitKey(0)

### 3. Find homgraphy
H = find_homography(points1, points2)

### 4. Combine images into a panorama

(size, offset) = calculate_size(img1, img2, H)
print ('output size: {0}  offset: {1}').format(size, offset)

panorama = merge_images(img1, img2, H, size, offset)
cv2.imwrite("panorama.jpg", panorama)
cv2.imshow('Panorama', panorama)
cv2.waitKey(0)


cv2.destroyAllWindows()
