# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
from firebase import firebase
import imutils
import time
import cv2
import os
import psycopg2
import datetime


savename=[]
car = 0
bus = 0
truck = 0
motorbike = 0

frames = 0

try:
    connection = psycopg2.connect(user = "erwin",
                                  password = "erwin123",
                                  host = "127.0.0.1",
                                  port = "5432",
                                  database = "postgres")

    cursor = connection.cursor()
	# print ( connection.get_dsn_parameters(),"\n")

    # Print PostgreSQL version
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("You are connected to - ", record,"\n")

except (Exception, psycopg2.Error) as error :
    print ("Error while connecting to PostgreSQL", error)

#connect to firebase
firebase = firebase.FirebaseApplication("https://vehicle-1d2dd.firebaseio.com/", None)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.join("/home/erwin/Pictures/PycharmProjects/yolo_image/yolo-coco", "coco.names")
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 4),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.join("/home/erwin/Pictures/PycharmProjects/yolo_image/yolo-coco", "yolov3.weights")
configPath = os.path.join("/home/erwin/Pictures/PycharmProjects/yolo_image/yolo-coco", "yolov3.cfg")

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture("cut.mp4")
writer = None
(W, H) = (None, None)

font = cv2.FONT_HERSHEY_SIMPLEX

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
	#initialize dateTime
	date_time = datetime.datetime.now()

	total_vehicle = []
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# frame = imutils.resize(frame,(640))

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	frame = adjust_gamma(frame, gamma=1.5)
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	center = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
		0.5)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			if LABELS[classIDs[i]] == "car" or LABELS[classIDs[i]] == "bus" or LABELS[classIDs[i]] == "truck" or LABELS[classIDs[i]] == "motorbike":
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
				cv2.putText(frame, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				# cv2.line(frame, (100,200), (350,200), [0, 255, 0], 2) 
				total_vehicle.append(LABELS[classIDs[i]])

				if LABELS[classIDs[i]] == "car" and y>600 and x>100 and x<1200:
					car = car+1
					LABELS[classIDs[i]] = ""
					postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence, weight) VALUES (%s,%s,%s,%s)"""
					record_to_insert = (str(date_time), "car", confidences[i], 100)
					cursor.execute(postgres_insert_query, record_to_insert)
					connection.commit()
					data = {
								"date_time": str(date_time),
								"vehicle_type": "car",
								"confidence": str(confidences[i]),
								"weight": str(100)
							}

					result = firebase.post("/vehicle-1d2dd/Costumer", data)

					print(result)

				if LABELS[classIDs[i]] == "bus" and y>600 and x>100 and x<1200:
					bus = bus+1
					LABELS[classIDs[i]] = ""
					postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence, weight) VALUES (%s,%s,%s,%s)"""
					record_to_insert = (str(date_time), "bus", confidences[i], 100)
					cursor.execute(postgres_insert_query, record_to_insert)
					connection.commit()
					data = {
								"date_time": str(date_time),
								"vehicle_type": "bus",
								"confidence": str(confidences[i]),
								"weight": str(100)
							}

					result = firebase.post("/vehicle-1d2dd/Costumer", data)
					print(result)

				if LABELS[classIDs[i]] == "truck" and y>600 and x>100 and x<1200:
					truck = truck+1
					LABELS[classIDs[i]] = ""
					postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence, weight) VALUES (%s,%s,%s,%s)"""
					record_to_insert = (str(date_time), "truck", confidences[i], 100)
					cursor.execute(postgres_insert_query, record_to_insert)
					connection.commit()
					data = {
								"date_time": str(date_time),
								"vehicle_type": "truck",
								"confidence": str(confidences[i]),
								"weight": str(100)
							}

					result = firebase.post("/vehicle-1d2dd/Costumer", data)
					print(result)

				if LABELS[classIDs[i]] == "motorbike" and y>600 and x>300 and x<1200:
					motorbike = motorbike+1
					LABELS[classIDs[i]] = ""
					postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence, weight) VALUES (%s,%s,%s,%s)"""
					record_to_insert = (str(date_time), "motorbike", confidences[i], 100)
					cursor.execute(postgres_insert_query, record_to_insert)
					connection.commit()
					data = {
								"date_time": str(date_time),
								"vehicle_type": "motorbike",
								"confidence": str(confidences[i]),
								"weight": str(100)
							}

					result = firebase.post("/vehicle-1d2dd/Costumer", data)
					print(result)
				
				## for case counting vehicle in parking
				# total_detect = len(total_vehicle)
				# car = len([p for p in total_vehicle if p == 'car'])
				# truck = len([p for p in total_vehicle if p == 'truck'])
				# bus = len([p for p in total_vehicle if p == 'bus'])
				# motorbike = len([p for p in total_vehicle if p == 'motorbike'])
				# bicycle = len([p for p in total_vehicle if p == 'bicycle'])
				
			cv2.rectangle(frame, (1200,20), (1400,120), (180, 132, 109), -1)
			cv2.putText(frame,'Car: '+ str(car),(1210, 40), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
			cv2.putText(frame,'Truck: '+ str(truck),(1210, 60), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
			cv2.putText(frame,'Bus: '+ str(bus),(1210, 80), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
			cv2.putText(frame,'motorbike: '+ str(motorbike),(1210, 100), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
			cv2.line(frame, (300,600), (1200,600), [0, 255, 0], 2) 

	else:
		cv2.rectangle(frame, (1200,20), (1400,120), (180, 132, 109), -1)
		cv2.putText(frame,'Car: '+ str(car),(1210, 40), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
		cv2.putText(frame,'Truck: '+ str(truck),(1210, 60), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
		cv2.putText(frame,'Bus: '+ str(bus),(1210, 80), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
		cv2.putText(frame,'motorbike: '+ str(motorbike),(1210, 100), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
		cv2.line(frame, (300,600), (1200,600), [0, 255, 0], 2) 


	## if you want to see window realtime video
	# cv2.imshow('vehicle counting',frame)
	# key = cv2.waitKey(1) & 0xFF
		
	# # if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	# 	break

	# if you want to write output video
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output_yolo.mp4", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)
	
		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)
	

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()