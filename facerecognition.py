import cv2

# Set input files
imagePath = "C://Users/gecete/Documents/ESTER/cascade/abba3.jpg"
cascPath = "C://Users/gecete/Documents/ESTER/cascade/lbpcascade_frontalface.xml"
#cascPath = "C://Users/gecete/Documents/ESTER/cascade/haarcascade_frontalface_default.xml"
cascEyePath = "C://Users/gecete/Documents/ESTER/cascade/haarcascade_eye.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(cascEyePath)
# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Detect eyes in the image
    roi_gray = gray[y:y + h, x:x + w]

    eyes = eyeCascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.3,
        minNeighbors=10,
        minSize=(30, 30)
    )
    for (x2, y2, w2, h2) in eyes:
        cv2.rectangle(image, (x+x2, y+y2), (x2+x + w2, y+y2 + h2), (0, 0, 255), 2)
cv2.imshow("Faces found", image)
cv2.waitKey(0)
