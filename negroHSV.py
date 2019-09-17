# USO
# python MEDIA.py --video videos.mov --output imagen.png
# C:\Python27\python.exe C:\Users\usuario\Desktop\iris\media.py -i cascada.mp4 -o gct17112017.png
import argparse
import numpy
import cv2

import os, sys
import numpy as np

# from scipy import stats


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
		img1 = cv2.resize(img1, (0, 0), fx=0.03, fy=0.03)
		if img1 is not None:
			images.append(img1)
	return images


# si paso un video
if ("." in args["input"]):
	vid_path = os.path.join(script_dir, args["input"])
	stream = cv2.VideoCapture(vid_path)
	r = 0
	numnew = 0
	frameBucket = []
	indi = 0
	framers = [2, 158, 200, 300, 400]
	while True:

		if indi == 5:
			break
		stream.set(cv2.CAP_PROP_POS_FRAMES, framers[indi] - 1)
		indi = indi + 1
		print("indi",indi)
		(grabbed, frame) = stream.read()

		if not grabbed:
			print("Fin")
			break
		numnew=numnew+1
		frameBucket.append(cv2.resize(frame, (0,0), fx=0.25, fy=0.25))


		# frameBucket.append(frame)
		# (B, G, R) = cv2.split(frame.astype("int"))
		r = r + 1

	transversal = []

	inc = 0
	taman = frameBucket[0].shape
	print("taman")
	print(taman)
	tol = taman[0] * taman[1]
	finalr = np.zeros((taman[0], taman[1]))
	finalg = np.zeros((taman[0], taman[1]))
	finalb = np.zeros((taman[0], taman[1]))

	i1 = 0
	while i1 < taman[0]:
		j = 0
		while j < taman[1]:
			transversalr = []
			transversalg = []
			transversalb = []
			imagen = 0
			# aswe=int(stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)*stream.get(cv2.cv.CV_CAP_PROP_FPS))

			frameBucket = np.asarray(frameBucket)

			while imagen < numnew:
				(rAvg1, gAvg1, bAvg1) = cv2.split(frameBucket[imagen].astype("int"))
				transversalr.append(rAvg1[i1][j])
				transversalg.append(gAvg1[i1][j])
				transversalb.append(bAvg1[i1][j])
				imagen = imagen + 1;
			repeticionesr = 999999
			repeticionesg = 999999
			repeticionesb = 999999
			datar = [];
			datar = transversalr;
			datab = [];
			datab = transversalb;
			datag = [];
			datag = transversalg;
			# print (data)
			for i in datar:
				aparece = datar.count(i)
				if aparece < repeticionesr:
					repeticionesr = aparece

			modar = []
			for i in datar:
				aparece = datar.count(i)
				if aparece == repeticionesr and i not in modar:
					modar.append(i)

			# print"modar:",i1,j,modar
			for i in datab:
				aparece = datab.count(i)
				if aparece < repeticionesb:
					repeticionesb = aparece

			modab = []
			for i in datab:
				aparece = datab.count(i)
				if aparece == repeticionesb and i not in modab:
					modab.append(i)

			# print "modab:" ,i1,j,modab

			for i in datag:
				aparece = datag.count(i)
				if aparece < repeticionesg:
					repeticionesg = aparece

			modag = []
			for i in datag:
				aparece = datag.count(i)
				if aparece == repeticionesg and i not in modag:
					modag.append(i)
			finalr[i1, j] = modar[0]
			finalb[i1, j] = modab[0]
			finalg[i1, j] = modag[0]
			print(tol)
			tol = tol - 1
			inc = inc + 1

			j = j + 1
		i1 = i1 + 1
















else:
	stream = load_images_from_folder(args["input"]);
	print("Total fotos: " + str(len(stream)))
	transversal = []
	inc = 0
	taman = stream[0].shape
	tol = taman[0] * 0.1 * 0.1 * taman[1]
	finalr = np.zeros((taman[0], taman[1]))
	finalg = np.zeros((taman[0], taman[1]))
	finalb = np.zeros((taman[0], taman[1]))

	i1 = 0
	while i1 < taman[0]:
		j = 0
		while j < taman[1]:
			transversalr = []
			transversalg = []
			transversalb = []
			imagen = 0
			while imagen < len(stream):
				(rAvg1, gAvg1, bAvg1) = cv2.split(stream[imagen].astype("int"))
				transversalr.append(rAvg1[i1][j])
				transversalg.append(gAvg1[i1][j])
				transversalb.append(bAvg1[i1][j])
				imagen = imagen + 15;
			repeticionesr = 0
			repeticionesg = 0
			repeticionesb = 0
			datar = [];
			datar = transversalr;
			datab = [];
			datab = transversalb;
			datag = [];
			datag = transversalg;
			# print (data)
			for i in datar:
				aparece = datar.count(i)
				if aparece > repeticionesr:
					repeticionesr = aparece

			modar = []
			for i in datar:
				aparece = datar.count(i)
				if aparece == repeticionesr and i not in modar:
					modar.append(i)

			# print"modar:",i1,j,modar
			for i in datab:
				aparece = datab.count(i)
				if aparece > repeticionesb:
					repeticionesb = aparece

			modab = []
			for i in datab:
				aparece = datab.count(i)
				if aparece == repeticionesb and i not in modab:
					modab.append(i)

			# print "modab:" ,i1,j,modab

			for i in datag:
				aparece = datag.count(i)
				if aparece > repeticionesg:
					repeticionesg = aparece

			modag = []
			for i in datag:
				aparece = datag.count(i)
				if aparece == repeticionesg and i not in modag:
					modag.append(i)

			# print "modag:" ,i1,j,modag
			finalr[i1, j] = modar[0]
			finalb[i1, j] = modab[0]
			finalg[i1, j] = modag[0]
			print(tol)
			tol = tol - 1
			inc = inc + 1
			j = j + 1
		i1 = i1 + 1

avg = cv2.merge([finalr, finalg, finalb]).astype("uint8")
cv2.imwrite(args["output"], avg)

if ("." in args["input"]):
	stream.release()