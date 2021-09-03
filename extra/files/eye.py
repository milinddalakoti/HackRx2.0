import cv2
import numpy as np

image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/Reading glasses.jpg')

grayimg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

eye_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(grayimg)
print(eyes)
for (x,y,w,h) in eyes:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),1)
print(eyes.shape[0])
cv2.imshow('Eyes',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#when eyes are clear it detects eyes properly i.e 2 otherwise it detects more eyes
'''
rects = detector.detectMultiScale(gray, scaleFactor=1.05,
	minNeighbors=7, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)
 use this to tune for proper detection of eyes
'''
