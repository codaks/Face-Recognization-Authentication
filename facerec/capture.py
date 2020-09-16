# -*- coding: utf-8 -*-
"""
Created on Mon May 18 23:28:21 2020

@author: piyus
"""
import cv2
import os
name=input("enter name")
# define the name of the directory to be created
path = "dataset/"+name

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
cam = cv2.VideoCapture(0)

path = "dataset/"+name+"/"
img_counter = 7

while img_counter!=0:
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

cv2.destroyAllWindows()
