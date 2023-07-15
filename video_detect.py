from flask import Flask, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load pretrained YOLOv8n models
shape_model = YOLO("./utils/yolo8n-shapes.pt", "v8")
gender_model = YOLO("./utils/yolov8-gender.pt", "v8")

# Function to read class data from file
def read_class_data(file_path):
    with open(file_path, "r") as file:
        class_list = file.read().splitlines()
    return class_list

# Read shape class data
shape_class_list = read_class_data("utils/coco.txt")

# Read gender class data
gender_class_list = read_class_data("utils/gender.txt")

# Open the camera for video capture
cap = cv2.VideoCapture(0)

@app.route('/video_detection', methods=['GET'])
def perform_detection():
    if not cap.isOpened():
        return jsonify({"error": "Cannot open camera"})

    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to read frame from camera"})

    # Predict on the frame
    shape_detect_params = shape_model.predict(source=[frame], show=True)
    gender_detect_params = gender_model.predict(source=[frame], show=True)

    results = []

    for shape_box in shape_detect_params[0].boxes:
        s_clsID = shape_box.cls.numpy()[0]
        s_cs = shape_class_list[int(s_clsID)]

        for gender_box in gender_detect_params[0].boxes:
            g_clsID = gender_box.cls.numpy()[0]
            g_cs = gender_class_list[int(g_clsID)]

            results.append({"gender": g_cs, "shape": s_cs})

    if len(results) == 0:
        return jsonify({"error": "Frame does not contain any detected objects. Kindly select a clearer frame for processing"}), 400

    return jsonify(results)

if __name__ == '__main__':
    app.run()
