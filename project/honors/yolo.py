# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
				dtype="uint8")

			# derive the paths to the YOLO weights and model configuration
			weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
			configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

			# load our YOLO object detector trained on COCO dataset (80 classes)
			print("[loading...........")
			net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

			# load our input image and grab its spatial dimensions
			image = cv2.imread(args["image"])
			(H, W) = image.shape[:2]
			cv2.imshow("Image1", image)
			print("here image show1")
			time.sleep(5)
			cv2.waitKey(1)
			if ord('q')==cv2.waitKey(10):
				exit(0)
			original = cv2.imread("images/photo.jpeg")
			duplicate = cv2.imread("images/photo2.jpeg")
		# 1) Check if 2 images are equals
			if original.shape == duplicate.shape:
				print("The images have same size and channels")
				difference = cv2.subtract(original, duplicate)
				b, g, r = cv2.split(difference)
				if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
					print("The images are completely Equal")
					cv2.imshow("Original", original)
					cv2.imshow("Duplicate", duplicate)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
				else :
					print("the images are not same")
					cv2.imshow("Original", original)
					cv2.imshow("Duplicate", duplicate)
					cv2.waitKey(0)
					cv2.destroyAllWindows()

			# round 2# initialize a list of colors to represent each possible class label
					# determine only the *output* layer names that we need from YOLO
					labelsPath = os.path.sep.join([args["yolo"], "coco.names"])

			imgResp=urllib.request.urlopen(url)
			imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
			img=cv2.imdecode(imgNp,-1)
			cv2.imwrite("images/photo2.jpeg",img)
			# initialize a list of colors to represent each possible class label
			np.random.seed(42)
			COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
				dtype="uint8")

			# derive the paths to the YOLO weights and model configuration
			weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
			configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

			# load our YOLO object detector trained on COCO dataset (80 classes)
			print("2.loading...........")
			net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

			# load our input image and grab its spatial dimensions
			image = cv2.imread(args["image"])
			(H, W) = image.shape[:2]
			cv2.imshow("Image2", image)
			print("here image show2")
			time.sleep(5)
			cv2.waitKey(1)
			if ord('q')==cv2.waitKey(10):
				exit(0)


			# determine only the *output* layer names that we need from YOLO
			ln = net.getLayerNames()
			ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			end = time.time()

			# show timing information on YOLO
			print("[INFO] YOLO took {:.6f} seconds".format(end - start))

			# initialize our lists of detected bounding boxes, confidences, and
			# class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > args["confidence"]:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping bounding
			# boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
				args["threshold"])

			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the image
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, color, 2)

			print("the number of bottles present",len(idxs))

			'''if(len(idxs)<=4):
				print("*********Action needed refill the stock********")
				sender = 'sandii68298@gmail.com'
				receivers = 'elsravi@gmail.com'
				s = smtplib.SMTP('smtp.gmail.com:587')
				s.starttls()
				s.login(sender,"password9100")
				message = """From: From Person <from@fromdomain.com>
				To: To Person <to@todomain.com>
				Subject: *****Sandeep project trial STOCK WARNING*****
				DEAR MATE.....REFILL THE RACK IT HAS ONLY.
				"""
				s.sendmail("sandii68298@gmail.com","elsravi@gmail.com", message)         
				print( "Successfully sent email")
				s.quit()'''
				

classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
