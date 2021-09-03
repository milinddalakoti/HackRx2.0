import cv2
import numpy as np

image = cv2.imread("/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/clear.png")
#print(image)
image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
grayImg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
'''
nose_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_mcs_nose.xml')
nose = nose_cascade.detectMultiScale(grayImg,1.7,11)
'''
mouth_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_smile.xml')
mouth = mouth_cascade.detectMultiScale(grayImg,1.7,11)
'''
for (x,y,w,h) in nose:
    y = int(y - 0.15*h)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),1)
'''

for (x,y,w,h) in mouth:
    y = int(y - 0.17*h)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),1)
    
print(mouth.shape[0])
cv2.imshow('Mouth and Nose',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
