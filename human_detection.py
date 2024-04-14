from ultralytics import YOLO
import cv2
import numpy as np
import urllib.request

#loading yolov8 modelgit remote add origin https://github.com/sunitagwachha/human-detection-using-smartphone.git
model =YOLO("yolov8n.pt")
model.predict(source= "img.JPG",show=True, save=True, conf =0.8)

model.predict(source=0, show =True, save=True, conf=0.8)


model.predict(source=0, show =True, save=True, conf=0.8, classes=[0,15] )

url ="http://192.168.1.220:8080/shot.jpg"

while True:
    image_source =np.ndarray(urllib.request.urlopen(url).read(),dtype ="utf-8")
    image_cv =cv2.imdecode(image_source,-1)
    final_img =cv2.resize(image_cv,(300,400))
    model.predict(source=0, show =True, save=True, conf=0.5, classes=[0])

    if ord =="q":
        break
    
cv2.destroyAllWindows()

model.predict(source=0, show =True, save=True, conf=0.8, classes=[0,15] )