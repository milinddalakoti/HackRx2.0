import cv2
import numpy as np 
from mtcnn.mtcnn import MTCNN
import dlib
from imutils import face_utils
import imutils
import reference_world as world
import numpy as np
import datetime

s_time = datetime.datetime.now()
img = cv2.imread('./images/professional.jpg')
#img = cv2.imread('./easy1.png')
#img = cv2.imread('./first.jpg')
#img = cv2.imread('./easy.jpg')
#img=cv2.imread('./multipleFace.png')


def mtc(img):
    """
    STEP 1 : Detecting face using Mutlti Convolutional Network

    Args:
        img (image): Input Image

    Returns:
        list : Returns the Region of Interest from image, Fitment Score and Criteria Status
    """
    
    fitment_score = -100
    detector1 = MTCNN()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces1 = detector1.detect_faces(img_rgb)
    print(faces1)
    if(len(faces1)==0):
        text="FAILED CRITERIA 1 : FACE NOT DETECTED"
        print(text)
        print("FITMENT SCORE : ",fitment_score)   
        return [img, fitment_score, text] 
    elif(len(faces1)>1):
        text="FAILED CRITERIA 1 : MULTIPLE FACES DETECTED"
        print(text)
        print("FITMENT SCORE : ",fitment_score) 
        return [img, fitment_score, text]
    else:
        #MTCNN
        for result in faces1:
            x, y, w, h = result['box']
            x1, y1 = x + w, y + h
            #cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
            x=x-25
            y=y-30
            x1=x1+40
            y1=y1+40
            roi = img[y:y1,x:x1]
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 255), 2)
            cv2.imwrite('./roi1.png',roi)
        fitment_score=fitment_score+30
        text="PASSED CRITERIA 1 : FACE DETECTED"
        print(text)
        #print("FITMENT SCORE : ",fitment_score) 
        cv2.imshow("mtcnn", roi)
        #print("Time Consumed MTCNN :",datetime.datetime.now()-s_time)
        return [roi, fitment_score, text]

def blur(colorImage, fitment_score):
    """
    STEP 2 : Detecting the amount of blur in the image

    Args:
        colorImage (image): Original Image
        fitment_score (int): Fitment Score of Image

    Returns:
        list: Returns the Region of Interest from image, Fitment Score and Criteria Status
    """
    
    grayImage=cv2.cvtColor(colorImage,cv2.COLOR_BGR2GRAY)
    laplace = cv2.Laplacian(grayImage, cv2.CV_64F).var()
    print(laplace)
    threshold = 55.0 
    if laplace < threshold:
        text="FAILED CRITERIA 2 : BLUR DETECTED"
        print(text)
        print("FITMENT SCORE : ",fitment_score)
        return [text ,fitment_score]
        
    else :
        fitment_score=fitment_score+30
        text="PASSED CRITERIA 2 : NO BLUR DETECTED"
        print(text)
        #print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]

def brightness(colorImage,fitment_score):
    """
    Checks the brightness of image 

    Args:
        colorImage (img): Input Image
        fitment_score (int): Fitment Score of Image

    Returns:
        list: Returns the Region of Interest from image, Fitment Score and Criteria Status
    """
    L, A, B = cv2.split(cv2.cvtColor(colorImage, cv2.COLOR_BGR2LAB))
    L = L/np.max(L)
    #print(np.mean(L))
    if(np.mean(L) < 0.4):
        text="FAILED CRITERIA 3 : DARK IMAGE"
        print(text)
        print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]
    
    elif(np.mean(L) > 1.0):
        text="FAILED CRITERIA 3 : TOO BRIGHT IMAGE"
        print(text)
        print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]
        
    else:
        fitment_score=fitment_score+30
        text="PASSED CRITERIA 3 : NORMAL IMAGE"
        print(text)
        #print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]
        
    
def eye(colorImage,fitment_score):
    """
    Detects Eyes in Image

    Args:
        colorImage (img): ROI image from face function
        fitment_score (int): Fitment Score of Image

    Returns:
        list: Returns the Region of Interest from image, Fitment Score and Criteria Status
    """
    grayImage = cv2.cvtColor(colorImage,cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(grayImage)
    #,scaleFactor=1.04,minNeighbors=13,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(eyes)==0:
        text="FAILED CRITERIA 4 : EYES NOT DETECTED"
        print(text)
        print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]

    elif eyes.shape[0]==2:
        fitment_score=fitment_score+30
        text="PASSED CRITERIA 4 : EYES DETECTED"
        print(text)
        #print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]
        
    else:
        text="FAILED CRITERIA 4 : EYES ARE COVERED"
        print(text)
        print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]
        
def mouth(colorImage,fitment_score): 
    """
    Checking if face is covered or not

    Args:
        colorImage (img): ROI Image from Face Function
        fitment_score (int): Fitment Score of Image

    Returns:
        list: Returns the Region of Interest from image, Fitment Score and Criteria Status
    """
    grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
    mouth_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_smile.xml')
    mouth = mouth_cascade.detectMultiScale(grayImage)
    #,scaleFactor=1.4,minNeighbors=26,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE) 
    countMouth = mouth.shape[0]
    if countMouth == 1:
        fitment_score=fitment_score+30
        text="Passed Criteria 5 : FACE NOT COVERED"
        print(text)
        #print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]
    
    else :
        text="FAILED CRITERIA 5 : FACE IS COVERED"
        print(text)
        print("FITMENT SCORE : ",fitment_score)
        return [colorImage,fitment_score,text]

def pose(colorImage,fitment_score):
    """
    Checking Head Pose in Image

    Args:
        colorImage (img): ROI Image from Face Function
        fitment_score (int): Fitment Score of Image

    Returns:
        list: Returns the Region of Interest from image, Fitment Score and Criteria Status
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    faces = detector(cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB), 0)
    face3Dmodel = world.ref3DModel()
    for face in faces:
        shape = predictor(cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB), face)
        refImgPts = world.ref2dImagePoints(shape)
        height, width, channel = colorImage.shape
        focalLength = 1 * width
        cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))
        mdists = np.zeros((4, 1), dtype=np.float64)
        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(
            face3Dmodel, refImgPts, cameraMatrix, mdists)
        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(
            noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)
        # draw nose line 
        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
        # calculating angle
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        x = np.arctan2(Qx[2][1], Qx[2][2])
        y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
        z = np.arctan2(Qz[0][0], Qz[1][0])
        gaze = "Looking: "
        if angles[1] < -15:
            text="WARNING CRITERIA 6 : SLIGHTLY LOOKING LEFT"
            print(text)
            print("FITMENT SCORE : ",fitment_score)
            return [colorImage,fitment_score,text]

        elif angles[1] > 15:
            text="WARNING CRITERIA 6 : SLIGHTLY LOOKING RIGHT"
            print(text)
            print("FITMENT SCORE : ",fitment_score)
            return [colorImage,fitment_score,text]
            
        else:
            fitment_score = fitment_score+30
            text="PASSED CRITERIA 6 : NORMAL IMAGE"
            print(text)
            print("FITMENT SCORE : ",fitment_score)
            return [colorImage,fitment_score,text]

if img is None:
    print("Empty Image // Image Not Found")
else:
    print("Original Height :", img.shape)
    img=cv2.resize(img,(350,350),interpolation=cv2.INTER_AREA)
    #img = cv2.resize(img, None, fx=2, fy=2)
    height, width = img.shape[:2]
    img1 = img.copy()
    list1 = mtc(img)
    afterMtc=list1[0]
    fitment_score = list1[1]
    blur(img1,fitment_score)
    brightness(img1,fitment_score)
    eye(afterMtc,fitment_score)
    mouth(afterMtc,fitment_score)
    pose(img1,fitment_score)
    
cv2.waitKey(0)
cv2.destroyAllWindows()


