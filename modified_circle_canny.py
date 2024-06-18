################################################
## ATTEMPT USING CV2 ONLY (FOR THE MOST PART) ##
## BUT ALSO USING SOME MORPHING OF IMAGES     ##
################################################

import cv2
import numpy as np

## LOAD MODEL -- CV2 USES ONNX MODELS FOR WHATEVER REASON ##
net = cv2.dnn.readNetFromONNX('yolov8-trained3.onnx')

## LOAD BASEBALL CLASS -- THERE SHOULD ONLY BE 1 ##
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

## OPEN VIDEO AND FIND PULL THE FOREGROUND FORward ##
video = cv2.VideoCapture('Baca-l_C_109618_HittingAssessment_ArizonaFacility_07-May-24_1715123617_H6_48_77.3.mov')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

frame_counter = 0

## PROCESS EACH FRAME -- LOOK FOR BALL ##
while video.isOpened():
    
    ret, frame = video.read()
    if not ret:
        break
    
    frame_counter += 1
    print(f"Processing frame {frame_counter}")
    
    ## BIG PUSH FOR GRAYSCALE ##
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## FORGET THE BACKGROUND ##
    fgmask = fgbg.apply(gray)

    ## WAYYYYY TO NOISY -- GET RID OF THIS ##
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    ## THE MOMENT OF TRUTH ##
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (1280, 1280), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    ## COUNTOURS -- BIG CIRCLE FINDER = BALL FINDER ##
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ## IGNORE THE AREAS NOT IN THE FOREGROUND ##
    circles = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            if len(scores) > 0:  # Check if scores is not empty
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.9 and classes[class_id] == 'baseball':
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    right = int(center_x + width / 2)
                    bottom = int(center_y + height / 2)

                    ## GET AREA OF INTREST ##
                    roi = fgmask[top:bottom, left:right]

                    ## LOOK FOR CIRCLE IN ROI ##
                    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        perimeter = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                        if len(approx) > 4 and len(approx) < 20:  # Filter based on shape complexity
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            if radius > 10 and radius < 50:  # Filter based on size
                                circles.append((int(x + left), int(y + top), int(radius)))

    ## CIRCLES AROUND THE CIRCLES (BALL) ##
    for circle in circles:
        x, y, radius = circle
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

## SHOW ME THE MONEY OR NOT ##
cv2.imshow('Circles', frame)

cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()