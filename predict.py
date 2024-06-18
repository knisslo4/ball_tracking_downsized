##################################################################
## LETS SEE HOW GOOD THIS MODEL REALLY IS AT FINDING A BASEBALL ##
##               1... 2... 3... PREDICT                         ##
##################################################################

from ultralytics import YOLO
import os

## LOAD MODEL -- YOLO BABY ##
model = YOLO('yolov8-trained2.pt')  # pretrained YOLOv8n model

## VIDEOS TO PROCESS -- THIS IS TEMPORARY AND NOT SCALABLE OF COURSE ##
video_files = ['/ball_tracking/predictions/videos/oData_in.MP4', '/ball_tracking/predictions/videos/oData_out.MP4']

## DETECT ##
for video_file in video_files:
    ## YOLO'S SAVING DEFAULTS ARE SUBPAR ##
    results = model.predict(source=video_file, save=True, conf=0.70)