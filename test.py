# Python3 program to extract and save video frame
# using OpenCV Python

#import computer vision module
import os
import cv2
import json
import logging
import pandas as pd

verbose = 1
# defines video path
video = 'p4r_arba.mp4'
# defines json path
file = open('p4r_arba.json')
data = json.load(file)

# capture the video
cap = cv2.VideoCapture(video)
i = 0  # frame index to save frames
jump = 100 # first valid index

while cap.isOpened():
    for descriptor in data:
        if "_" not in descriptor:
            if verbose: print("descriptor:", descriptor)
            for coordinate in data[descriptor]:
                path = f"images/{descriptor}/{coordinate}"
                if not os.path.exists(path): os.makedirs(path)
                if verbose: print("coordinate:", coordinate)
                for value in data[descriptor][coordinate]:
                    if verbose: print("value:", value)
                    if value >= jump:
                        if verbose: print("valid frame value")
                        cap.set(1,int(value))
                        ret, frame = cap.read()
                        if ret:
                            if verbose: print("writing frame number:",value)
                            cv2.imwrite(f"{path}/{value}.png", frame)

