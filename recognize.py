# USAGE
# python recognize.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
                   

userid="piyush"
cam = cv2.VideoCapture(0)
path = "output/"
img_counter = 1

while img_counter!=0:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    # SPACE pressed
    img_counter =img_counter- 1

    protoPath = "face_detection_model\\deploy.prototxt"
    modelPath = "face_detection_model\\res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    
    # load our serialized face embedding model from disk
    
    embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
    
    
    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
    le = pickle.loads(open( "output/le.pickle", "rb").read())
    
    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    image = frame
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
    	cv2.resize(image, (300, 300)), 1.0, (300, 300),
    	(104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    login=0
    # loop over the detections
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
    	# prediction
    	confidence = detections[0, 0, i, 2]
    
    	# filter out weak detections
    	if confidence > 0.5:
    		# compute the (x, y)-coordinates of the bounding box for the
    		# face
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")
    
    		# extract the face ROI
    		face = image[startY:endY, startX:endX]
    		(fH, fW) = face.shape[:2]
    
    		# ensure the face width and height are sufficiently large
    		if fW < 20 or fH < 20:
    			continue
    
    		# construct a blob for the face ROI, then pass the blob
    		# through our face embedding model to obtain the 128-d
    		# quantification of the face
    		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
    			(0, 0, 0), swapRB=True, crop=False)
    		embedder.setInput(faceBlob)
    		vec = embedder.forward()
    
    		# perform classification to recognize the face
    		preds = recognizer.predict_proba(vec)[0]
    		j = np.argmax(preds)
    		proba = preds[j]
    		name = le.classes_[0]
    		
    		if name==userid:
    			login=1
    			print("login")
    		# draw the bounding box of the face along with the associated
    		# probability
    		text = "{}: {:.2f}%".format(name, proba * 100)
    		y = startY - 10 if startY - 10 > 10 else startY + 10
    		cv2.rectangle(image, (startX, startY), (endX, endY),
    			(0, 0, 255), 2)
    		cv2.putText(image, text, (startX, y),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
cam.release()

cv2.destroyAllWindows()
    
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)