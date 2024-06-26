# BALL TRACKING
This is a computer vision model that is trained through YOLO framework to detect a baseball throughout frames. It's been trained on 11k+ labeled/unlabeled images of behind pitcher view (pitch design view). This project would be for anyone looking to get into computer vision and sports analytics. Much of the project has been spent training the model to actually detect a ball and do so decently well, therefore no analysis of the data has been started.

## Project History
- Originally the model was just trained on images to place a bounding box around a baseball
- Taken steps after initial:
  detecting a baseball from a hitter's side view using only physics and Hugh Circles/edge detection
  - had some success here but it's difficult to just isolate specific areas
 NEXT STEPS:
- perform analysis on the data; estimate pitch velocity*
- increase the data set with edgertronic cameras to define seam orientation; estimating spin components of a pitch

<p align="center">
  <img src="mlb.gif" alt="MLB Demo" width="500"/>
  <img src="outside.gif" alt="Outside Demo" width="500"/>
  <img src="indoor.gif" alt="Indoor Demo" width="500"/>
</p>
