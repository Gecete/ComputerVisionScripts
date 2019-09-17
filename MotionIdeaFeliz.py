import argparse
import imutils
import cv2
import os, sys
import numpy as np
from matplotlib import pyplot as plt
#C:\Python27\python.exe C:\Users\gecete\PycharmProjects\diarization\MotionIdeaFeliz.py -i C:\Users\gecete\Documents\iris\zapas  -o zapabuean.png

script_dir = sys.path[0]

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
				help="path to input video or image files")
ap.add_argument("-o", "--output", required=True,
				help="path to output 'long exposure'")
args = vars(ap.parse_args())

(rAvg, gAvg, bAvg) = (None, None, None)
total = 0


def load_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		print(filename)
		img1 = cv2.imread(os.path.join(folder, filename))
		img = img1  # cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
		if img is not None:
			images.append(img)
	return images


if ("." in args["input"]):
	vid_path = os.path.join(script_dir, args["input"])
	stream = cv2.VideoCapture(vid_path)

	inicio = 657 * 60
	fin = inicio + 1


	frame_no = inicio
	total = inicio
	stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
	(grabbed, frame) = stream.read()

	(Bi, Gi, Ri) = cv2.split(frame.astype("float"))
	while True:
		(grabbed, frame) = stream.read()

		if not grabbed:
			print("Fin")
			break

		if total%10 == 0:
			# frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
			(B, G, R) = cv2.split(frame.astype("float"))

			if rAvg is None:
				rAvg = Ri
				bAvg = Bi
				gAvg = Gi

			else:
				length = stream.get(cv2.CAP_PROP_FRAME_COUNT)
				print(total - fin)
				sys.stdout.write('\r' + str(total * 100 / length) + "% Loading  ")
				(m, n) = rAvg.shape
				for i in xrange(m):
					for j in xrange(n):
						if rAvg[i][j] > Ri[i][j]:
							rAvg[i][j] = R[i][j]
						if bAvg[i][j] > Bi[i][j]:
							bAvg[i][j] = B[i][j]
						if gAvg[i][j] > Gi[i][j]:
							gAvg[i][j] = G[i][j]

		total += 1
		if total == fin:
			break
else:  # imagenes
	stream = load_images_from_folder(args["input"]);
	print("Total fotos: " + str(len(stream)))
	imgq = cv2.imread('www.jpg')
	hsv = cv2.cvtColor(imgq, cv2.COLOR_BGR2HSV)

	(Bi, Gi, Ri) = cv2.split(hsv.astype("float"))
	y, x, _ = plt.hist(np.ndarray.flatten(Bi), bins=180, density=True)

	matriz = [y, x]
	elem = np.argsort(y)
	print(elem[-1], elem[-2])
	y[::-1].sort()
	print(y[0], y[1])
	total=0
	np.shape(Bi)
	rAvgG = np.zeros(np.shape(Bi))
	bAvgG = np.zeros(np.shape(Bi))
	gAvgG = np.zeros(np.shape(Bi))
	for imagen in stream:
		# imagen = cv2.resize(imagen, (0,0), fx=0.5, fy=0.5)
		hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
		(B, G, R) = cv2.split(hsv.astype("float"))
		height, width, channels = hsv.shape;
		s = (height, width);
		print(total)

		if rAvg is None:
			rAvg = R
			bAvg = B
			gAvg = G

		else:
			(m, n) = rAvg.shape
			for i in xrange(m):
				for j in xrange(n):
					if  np.abs(bAvg[i][j]-elem[-1])>3:
						Ri[i][j] = rAvg[i][j]
						Bi[i][j] = bAvg[i][j]
						Gi[i][j] = gAvg[i][j]

		total += 1

avg = cv2.merge([Bi, Gi, Ri]).astype("uint8")

y, x, _ = plt.hist(np.ndarray.flatten(Bi), bins=180, density=True)

elem = np.argmax(y)
elem2 = np.argmax([y[1:np.size(y)-1]])
print(elem, elem2)
y[::-1].sort()
elem = np.argmax(y)
elem2 = np.argmax([y[1:np.size(y)-1]])
print(y[0],y[1])
print(elem,elem2)
plt.show()


avg = cv2.cvtColor(avg, cv2.COLOR_HSV2BGR)

cv2.imwrite(args["output"], avg)

if ("." in args["input"]):
	stream.release()