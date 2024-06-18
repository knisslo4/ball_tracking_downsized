############################################################################
## BECAUSE YOLO OUTPUTS AN AVI FILE AND I HAVEN'T FOUND A WAY AROUND THIS ##
############################################################################

import os
import cv2

##########
## MAIN ##
##########

def convert(file):
    output_folder = 'predictions/output_videos'
    input_folder = '/ball_tracking/runs/detect/predict2'
    os.makedirs(output_folder, exist_ok=True)
    
    file_path = os.path.join(input_folder, file)
    print(f"Converting file: {file_path}")
    
    cap = cv2.VideoCapture(file_path)

    ## MAKE SURE VIDEO STUFF LINES UP ##
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    index = 3
    
    file_name = os.path.basename(file)
    mp4_file_name = os.path.splitext(file_name)[0] + f'_result{index}.mp4'
    mp4_file_path = os.path.join(output_folder, mp4_file_name)
    print(f"Output file: {mp4_file_path}")

    
    out = cv2.VideoWriter(mp4_file_path, codec, fps, (width, height))

    ## MP4 !!!!!!!!!!! ##
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    
    index =+ 1
    
    print(f"Conversion complete for file: {file_path}")
    return mp4_file_path

## ALWAYS HAVE TO ADJUST THIS BUT OH WELL ##
input_folder = '/ball_tracking/runs/detect/predict2'

files = os.listdir(input_folder)

for file in files:
    extension = os.path.splitext(file)[1].lower()
    if extension != ".mp4":
        convert(file)
    else:
        print(f"{file} is already an mp4")
        

        