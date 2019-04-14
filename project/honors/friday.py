# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import smtplib
import time
import cv2
import os
import urllib.request
from PIL import Image
from pytesseract import image_to_string
from gtts import gTTS 
import os
from sinchsms import SinchSMS
import pytesseract
# construct the argument parse and parse the arguments
#url='http://192.168.43.1:8080/photo.jpg'
url='http://192.168.43.1:8080/photo.jpg'
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=False,
	#help="images/shot.jpg")
#ap.add_argument("-y", "--yolo", required=False,
	#help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


######
args["yolo"]="yolo-coco"
args["image"]="images/photo.jpeg"
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

while True:
#for i in range(3):
	print("first_snap")
	imgResp=urllib.request.urlopen(url)
	imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
	img=cv2.imdecode(imgNp,-1)
	cv2.imwrite("images/photo.jpeg",img)
	# load our input image and grab its spatial dimensions
	image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]
	print("here image show1")
	#cv2.imshow("Image1", image)
	#time.sleep(3)
	#cv2.destroyAllWindows()
	cv2.waitKey(1)
	if ord('q')==cv2.waitKey(10):
		exit(0)
	# round 2
	print("second snap")
	time.sleep(6)
	imgResp=urllib.request.urlopen(url)
	imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
	img=cv2.imdecode(imgNp,-1)
	cv2.imwrite("images/photo2.jpeg",img)
	image = cv2.imread(args["image"])
	(H, W) = image.shape[:2]
	print("here image show2")
	#cv2.imshow("Image2", image)
	#time.sleep(3)
	#cv2.destroyAllWindows()
	cv2.waitKey(1)
	if ord('q')==cv2.waitKey(10):
		exit(0)
	frame1 = cv2.imread("images/photo.jpeg")
	frame2 = cv2.imread("images/photo2.jpeg")
	grey1=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) 
	grey2=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	#normalizedImg1= np.zeros((800, 800))
	#normalizedImg1= cv.normalize(grey1,  normalizedImg1, 0, 255, cv.NORM_MINMAX)
	#normalizedImg2 = np.zeros((800, 800))
	#normalizedImg2= cv.normalize(grey2,  normalizedImg2, 0, 255, cv.NORM_MINMAX)
	#grey1 = normalize(img1)
    #grey2 = normalize(img2)
# 1) Check if 2 images are equals
	#time.sleep(3)
	'''cv2.imshow("grey1",grey1)
	cv2.waitKey(1)
	if ord('q')==cv2.waitKey(10):
		exit(0)
	#time.sleep(2)
	cv2.imshow("grey2",grey2)
	cv2.waitKey(1)
	if ord('q')==cv2.waitKey(10):
		exit(0)'''

	if grey1.shape == grey2.shape:
		print("The images have same size and channels")
		#time.sleep(3)
		#cv2.imshow("grey1",grey1)
		#time.sleep(2)
		#cv2.waitKey(1)
		#if ord('q')==cv2.waitKey(10):
		#	exit(0)
		#cv2.imshow("grey2",grey2)
		#cv2.waitKey(1)
		#if ord('q')==cv2.waitKey(10)

		#	exit(0)
		print("grey1",grey1)
		print("grey2",grey2)
		diff= cv2.subtract(grey1, grey2)
		print("difference ",diff)
		#diff =cv2.normalize(diff_un, None)
		#print("difference normalized",diff)
	
		#b, g, r = cv2.split(difference)
		#if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
		#if len(difference)-np.count_nonzero(difference) > np.count_nonzero(difference)  :
		x=np.sum(diff)
		print(x)
		if x>2650000:
			print("The images are not completely equal")
			#cv2.imshow("frame1", frame1)
			print("here after 30 seconds the changed frame image is taken and sent to the processing")

			time.sleep(5)

			imgResp=urllib.request.urlopen(url)
			imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
			img=cv2.imdecode(imgNp,-1)
			cv2.imwrite("images/photo.jpeg",img)
			image = cv2.imread(args["image"])
			(H, W) = image.shape[:2]
			#cv2.imshow("frame2", frame2)
			#cv2.waitKey(10)
			#cv2.destroyAllWindows()
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
			count_bottle=0
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					
					# draw a bounding box rectangle and label on the image
					color = [int(c) for c in COLORS[classIDs[i]]] 
					if(LABELS[classIDs[i]]=="bottle" ) :	
						cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
						text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
						cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
							0.5, color, 2)
						count_bottle=count_bottle+1


				print("near coordinates display Area")
				cv2.imshow("Image", image)
				time.sleep(5)
				cv2.waitKey(1)
				if ord('q')==cv2.waitKey(10):
					exit(0)

			def ocrs():
				image = cv2.imread("images/photo.jpeg")
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]	
				gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
				filename = "{}.png".format("temp")
				cv2.imwrite(filename, gray)
				text = pytesseract.image_to_string(Image.open(filename))
				print("from the image text is",text)		
				if (len(idxs)<=2) :
					print("*********Action needed refill the stock********")
					print("the number of bottles present from sum",count_bottle)	
					k=len(idxs)
					number = '+918919134330'
					#message = 'Warning! Less number of bottles in the rack '+ text +' only ' + str(k) + ' bottles are left please refill the rack!.' 
					message = 'Warning! Less number of bottles in the rack  only ' + str(k) + ' bottles are left please refill the rack!.'
					def sms_send(number,message):
					    client = SinchSMS('df5a3b30-0728-403f-9ccf-1efa4e570bd7','BLDUeKvHvEWbkcpnLId6AA==')

					    print("Sending '%s' to %s" % (message, number))
					    response = client.send_message(number, message)
					    message_id = response['messageId']

					    response = client.check_status(message_id)
					    while response['status'] != 'Successful':
					        print(response['status'])
					        time.sleep(1)
					        response = client.check_status(message_id)
					        print(response['status'])
					sms_send(number,message)

			try:
			    ocrs()
			except Exception as e:
			    print(e.args) 
			    print(e.__cause__)
			
			#myobj = gTTS(text=em, lang=language, slow=False) 
			#engine.say(myobj) 
			
			print("the total number of objects in the frame",len(idxs)) 

		else :
			print("the images are same")
			print("going to next iter in 5 second")
			time.sleep(3)
			cv2.imshow("frame1", frame1)
			cv2.imshow("frame2", frame2)
			cv2.waitKey(10)
			if ord('q')==cv2.waitKey(10):
					exit(0)
			cv2.destroyAllWindows()

	

	




