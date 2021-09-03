import numpy as np
import cv2
import dlib
import imutils
import time

class Detection():
    def __init__(self):
        self.image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/images/face1.png')
        self.face_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_frontalface_default.xml')
    
    def face(self):
        #face_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_frontalface_default.xml')
        #image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/images/multiface.png')
        #image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/images/face1.png')
        grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
  
        faces = self.face_cascade.detectMultiScale(grayImage)
  
        print(type(faces))
        countFace = faces.shape[0]
        
        if len(faces) == 0:
            print("No faces found")
  
        elif(countFace==1):
    
            #print(faces)
            #print(faces.shape)
            #print("Number of faces detected: " + str(faces.shape[0]))
    
            #countFace = faces.shape[0]
            #print(type(countFace))
            #if countFace==1:
            print("One Face Detected!! \n Selecting ROI")
                #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                #faces = faces.detectMultiScale(grayImage,1.3,5)
                
            for (x,y,w,h) in faces:
                x=x-150
                y=y-200
                w=w+300
                h=h+300
                roi_gray = grayImage[y:y+h,x:x+w]
                roi_color = self.image[y:y+h,x:x+w]
            cv2.imshow('ROI',roi_color)
        else:
            print("More than one face detected")
                
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = Detection()
    obj.face()
