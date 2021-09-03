# HackRx2.0
Face Detection using OpenCV

Our Solution:
We are using openCV and Haar cascades to detect particular features of a face. 
We have set several thresholds that check for the above mentioned features. 
As the image clears the threshold, the fitment score of the image keeps updating. 
If the images passes all the thresholds, it would have a positive score. If the image, at any step, gets rejected,
it would have a negative score, unless it passes the sanity check, in which case, it’s Fitment score would be zero.

Tech Used:
Back-End:
Python
Flask
NodeJS
OpenCV
DLib
Imutils
Front-End:
HTML
CSS
NodeJS
Bootstrap
Cloud Service Providers
Heroku
