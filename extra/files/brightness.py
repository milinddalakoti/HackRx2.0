#importing modules
import cv2
import time
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread("home/pradyumn/scripts/hackathons/Hackrx/main/backend/test pics/Lighting 1.jpg")
#img = cv2.imread("./images/original.png")
#image = cv2.resize(img, (10,10))
# Convert color space to LAB format and extract L channel
L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
# Normalize L channel by dividing all pixel values with maximum pixel value
L = L/np.max(L)
print(np.mean(L))
# Return True if mean is greater than thresh else False
if(np.mean(L) < 0.5):
    print("Dark")
elif(np.mean(L) > 1.0):
    print('Too bright')
else:
    print('Normal')    

labim = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
cv2.imwrite('./imgLAB.jpg',labim)
#cv2.imshow("LAB img",labim)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.putText(img, "{}".format(text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
