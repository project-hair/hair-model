from ultralytics import YOLO
import numpy

# load a pretrained YOLOv8n model
model = YOLO("yolov8n-seg-custom.pt", "v8")

# predict on an image
detection_output = model.predict(source="img.jpg", show=True)

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())
