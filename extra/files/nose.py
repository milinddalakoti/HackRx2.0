import cv2
import numpy as np

nose_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_nose.xml')

if nose_cascade.empty():
  raise IOError('Unable to load the nose cascade classifier xml file')

ds_factor = 0.5

image = cv2.imread("/home/pradyumn/scripts/hackathons/Hackrx/images/face1.png")
frame = cv2.resize(image, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in nose_rects:
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    break
cv2.imshow('Nose Detector', frame)

c = cv2.waitKey(0)


cv2.destroyAllWindows()