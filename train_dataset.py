##################################################
## TRAIN A MODEL TO DEECT A BASEBALL USING YOLO ##
##################################################

from ultralytics import YOLO
import torch
import traceback

##########
## MAIN ##
##########

def main():
    ## SET UP GPU ##
    cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f"{len(cuda_devices)} GPU devices available")
    results = None
    if len(cuda_devices) >= 1:
        device=0
    else:
        device='cpu'
    results = None

    ## LOAD MODEL ##
    model = YOLO('yolov8-trained2.pt')

    ## TRAIN MODEL ##
    try:
        results = model.train(
            data='train_dataset.yaml',
            epochs=200,
            patience=15,
            imgsz=640,
            batch=16,
            device=[0]
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    if results:
        print(results)
    else:
        print("Training unsuccessful :(")

if __name__ == '__main__':
    main()