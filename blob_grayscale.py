#####################################################################
## MAKING MODIFICATIONS TO IMAGES BEFORE RUNNING INFERENCE ON THEM ##
##   TRYING GRAYSCALE FIRST BUT DIDN'T MAKE MUCH OF A DIFFERENCE   ##
#####################################################################

import cv2
import numpy as np

## OPEM THE VIDEO ##
cap = cv2.VideoCapture('Baca-l_C_109618_HittingAssessment_ArizonaFacility_07-May-24_1715123617_H6_48_77.3.mov')

## FIND AREA TO LOOK FOR MOVING OBJECT ##
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.maxArea = 1000
params.filterByCircularity = True
params.minCircularity = 0.7
params.filterByConvexity = True
params.minConvexity = 0.8

## LOAD AREA OF INTEREST ##
detector = cv2.SimpleBlobDetector_create(params)

## FOR EACH FRAME ##
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    ## CONVERT FRAME TO GRAYSCALE ##
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## DETECT CIRCLE ##
    keypoints = detector.detect(gray)

    ## DRAW AROUND CIRCLE = BALL!! -- HOPEFULLY ##
    frame_with_blobs = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ## SHOW ME THE BLOBS ##
    cv2.imshow('Frame', frame_with_blobs)

    ## CLOSE OUT DISPLAY ##
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()