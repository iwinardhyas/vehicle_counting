import numpy as np
import argparse
import time
import cv2
import os
import json
import sys
import zmq


context = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5558")

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


labelsPath = os.path.join("model", "label.names")
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.join("model", "model_vehicle.weights")
configPath = os.path.join("model", "model_vehicle.cfg")


print("[INFO] loading model from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture('videos/bandung_road5.mp4')

W = None
H = None
font = cv2.FONT_HERSHEY_SIMPLEX

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1


while True:
	data = receiver.recv()
	total_vehicle = []
	(grabbed, frame) = vs.read()
	frame = cv2.resize(frame,(640,480))

	if not grabbed:
		break

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > args["confidence"]:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			if LABELS[classIDs[i]] == "motorbike" or LABELS[classIDs[i]] == "bicycle" \
			or LABELS[classIDs[i]] == "car" or LABELS[classIDs[i]] == "bus" or LABELS[classIDs[i]] == "truck":

				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
				text = "{}".format(LABELS[classIDs[i]])
				cv2.putText(frame, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
				total_vehicle.append(LABELS[classIDs[i]])

	####### IF YOU WANT TO SAVE AS MP$ check if the video writer is None
	# if writer is None:
	# 	# initialize our video writer
	# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	# 	writer = cv2.VideoWriter(args["output"], fourcc, 30,
	# 		(frame.shape[1], frame.shape[0]), True)

	total_detect = len(total_vehicle)
	car = len([p for p in total_vehicle if p == 'car'])
	truck = len([p for p in total_vehicle if p == 'truck'])
	bus = len([p for p in total_vehicle if p == 'bus'])
	motorbike = len([p for p in total_vehicle if p == 'motorbike'])
	bicycle = len([p for p in total_vehicle if p == 'bicycle'])
	
	cv2.rectangle(frame, (10, 275), (300, 400), (180, 132, 109), -1)
	cv2.putText(frame,'JALAN RAYA PASIR KOJA: ',(10, 285), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame, str(data),(10, 300), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame,'Detected Vehicles: '+ str(total_detect),(10, 315), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame,'Car: '+ str(car),(10, 330), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame,'Truck: '+ str(truck),(10, 345), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame,'Bus: '+ str(bus),(10, 360), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame,'motorbike: '+ str(motorbike),(10, 375), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)
	cv2.putText(frame,'Bicycle: '+ str(bicycle),(10, 390), font,0.5,(0, 0xFF, 0xFF),1,cv2.FONT_HERSHEY_SIMPLEX)

	data = {
		"street": 'JALAN RAYA PASIR KOJA KOTA BANDUNG',
		"sensors": [str(data)],
		"vehicle_detection": [{"total_detect": total_detect,
		"car": car,
		"truck": truck,
		"bus": bus,
		"motorbike": motorbike,
		"bicycle": bicycle}]
	}

	print(json.dumps(data, indent=4))
	cv2.imshow("Preview",frame)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break

vs.release()
