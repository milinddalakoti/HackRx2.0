import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('/home/pradyumn/scripts/hackathons/Hackrx/haarcascade_frontalface_default.xml')
  
#image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/images/multiface.png')
#image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/images/face1.png')
image = cv2.imread('/home/pradyumn/scripts/hackathons/Hackrx/main/backend/Image Dataset/Mask.png')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
faces = face_cascade.detectMultiScale(grayImage)
  
print(type(faces))
  
if len(faces) == 0:
    print("No faces found")
  
else:
    
    print(faces)
    print(faces.shape)
    print("Number of faces detected: " + str(faces.shape[0]))
    
    countFace = faces.shape[0]
    #print(type(countFace))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        print(x,y,w,h)
    x=x-150
    y=y-200
    w=w+300
    h=h+300
    
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)    
    
    #cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
    #cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
  
    cv2.imshow('Image with faces',image)
    
    
    if countFace==1:
        print("One Face Detected!! /n Selecting ROI")
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #faces = faces.detectMultiScale(grayImage,1.3,5)
        
        for (x,y,w,h) in faces:
            x=x-150
            y=y-200
            w=w+300
            h=h+300
            roi_gray = grayImage[y:y+h,x:x+w]
            roi_color = image[y:y+h,x:x+w]
        
        
        cv2.imshow('ROI',roi_color)
    else:
        print("More than one face detected")
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
