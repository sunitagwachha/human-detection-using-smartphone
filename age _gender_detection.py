import numpy as np
import cv2

age_list=[(0-10),(10-20),(20-30),(30-40),(40-50),(50-60),(60-70)]
gender_list =["male","female"]

model =YOLO("yolov8n.pt")
face=model.predict(source= "img.JPG",show=True, save=True, conf =0.8)

age_net.caffemodel and deploy_age.prototxt


