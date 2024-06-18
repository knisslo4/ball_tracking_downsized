import torch
import convert_model
'''
# Load your PyTorch model
model = torch.load('yolov8-trained3.pt')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 416, 416)

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, 'ball_model.onnx', verbose=True)
'''

from ultralytics import YOLO

model = YOLO('yolov8-trained3.pt')


model.export(format='onnx') 
