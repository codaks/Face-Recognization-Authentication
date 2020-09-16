from django.shortcuts import render,redirect
import cv2
import os
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
from .models import Login
from django.http import StreamingHttpResponse
import base64
from django.http import JsonResponse,HttpResponse
from io import BytesIO
from PIL import Image
import re
import base64
from django.contrib import messages

# Create your views here.

def login(request):
    if request.method == "POST":
        print("##########################################################")
        image_data = request.POST['img_data']
        image_data = re.sub("^data:image/png;base64,", "", image_data)
        image_data = base64.b64decode(image_data)
        image_data = BytesIO(image_data)
        im = Image.open(image_data)
        rgb_im = im.convert('RGB')
        
        m1 = rgb_im.save("geeks_new.jpg") 
        
        rgb_im = np.array(im)
        open_cv_image =rgb_im[:, :, ::-1].copy() 

        userid = request.POST['usr']
        pas = request.POST['pass']
        
        path = "output/"
        img_counter = 1

        while img_counter!=0:
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
            image = cv2.imread("geeks_new.jpg")
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            print("**********Inside Loop**********")
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
                print("**********Inside For Loop**********")
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
            
                # filter out weak detections
                print("**********Outside conf**********",confidence)
                if confidence > 0.5:
                    print("**********Inside Conf**********",confidence)
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
                    print("**********Predection**********")
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]
                    print("########################################################")
                    print(name)
                    print("########################################################")
                    if name==userid:
                        login=1
                        try:
                            user = Login.objects.get(username=userid,password=pas)
                            if user is not None:
                                data = {
                                'Login': True,'Status':"400",
                                'msg': 'You are Loggedin'
                                }

                                return redirect("/user/"+name)
                        except:
                            messages.info(request,"User Not Found")
                            return redirect("/")
                    else:
                        
                        messages.info(request,"Face Not Matched")
                        return redirect("/")
                    # draw the bounding box of the face along with the associated
            
                    # probability'''
            return redirect("/")
        return redirect("/")
    
    else:
        return render(request,"login.html")
        

def testing(request):
    if request.method=="POST":
        print("#################################################################################")
        image_data = request.POST['img_data']
        image_data = re.sub("^data:image/png;base64,", "", image_data)
        image_data = base64.b64decode(image_data)
        image_data = BytesIO(image_data)
        im = Image.open(image_data)
        rgb_im = im.convert('RGB')
        m1 = rgb_im.save("geeks.jpg") 
        print("################################")
        data = {
        'is_taken': True
        }
        return JsonResponse(data)
    else:
        return render(request,"testing.html")


def user(request,name):
    return render(request,"index.html",{'name':name})

def register(request):
    if request.method == "GET":
        
        
        return render(request,"register.html")
    else:
        print("####################################################################################")
        print("####################################################################################")
        name= request.POST['usr']
        pas = request.POST['pass']
        r_name = request.POST['name']

        ####################################################################################
        ######################   THIS IS ONLY FOR SAVING IMAGES     ########################
        ####################################################################################
        path ='dataset/'+name
        os.mkdir(path)
        #return redirect("/")
        for count in range(1,11):
            count = str(count)
            image_data = request.POST[count]
            image_data = re.sub("^data:image/png;base64,", "", image_data)
            image_data = base64.b64decode(image_data)
            image_data = BytesIO(image_data)
            im = Image.open(image_data)
            rgb_im = im.convert('RGB')
            path = "dataset/"+name+"/opencv_frame_"+count+".jpg"
            m1 = rgb_im.save(path) 
        

        ####################################################################################
        #####################################   END     ####################################
        ####################################################################################

        # define the name of the directory to be created
        path = "dataset/"+name

        '''try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)
        cam = cv2.VideoCapture(0)

        path = "dataset/"+name+"/"
        img_counter =10'''

        '''while img_counter!=0:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            des=path+img_name
            cv2.imwrite(des, frame)
            print("{} written!".format(img_name))
            img_counter =img_counter- 1

        cam.release()

        cv2.destroyAllWindows()'''


        # load our serialized face detector from disk

        protoPath = "face_detection_model\\deploy.prototxt"
        modelPath = "face_detection_model\\res10_300x300_ssd_iter_140000.caffemodel"
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load our serialized face embedding model from disk

        embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

        # grab the paths to the input images in our dataset

        imagePaths = list(paths.list_images("dataset"))

        # initialize our lists of extracted facial embeddings and
        # corresponding people names
        knownEmbeddings = []
        knownNames = []

        # initialize the total number of faces processed
        total = 0

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
            
            name = imagePath.split(os.path.sep)[-2]

            # load the image, resize it to have a width of 600 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            image = cv2.imread(imagePath)
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

            # ensure at least one face was found
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI and grab the ROI dimensions
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # add the name of the person + corresponding face
                    # embedding to their respective lists
                    knownNames.append(name)
                    knownEmbeddings.append(vec.flatten())
                    total += 1
            else:
                return redirect("/")
        # dump the facial embeddings + names to disk
        print("[INFO] serializing {} encodings...".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open("output/embeddings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()

        # load the face embeddings
        print("[INFO] loading face embeddings...")
        data = pickle.loads(open("output\\embeddings.pickle", "rb").read())

        # encode the labels
        print("[INFO] encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        # write the actual face recognition model to disk
        f = open("output\\recognizer.pickle", "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        f = open("output\\le.pickle", "wb")
        f.write(pickle.dumps(le))
        f.close()


        #cam.release()
        #cv2.destroyAllWindows()
        user = Login()
        user.name =r_name
        user.username = request.POST['usr']
        user.password = pas

        user.save()
        return redirect("/")