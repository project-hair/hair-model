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

@app.route('/image_detection', methods=['POST'])
def perform_detection():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"})

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Predict on the image
    shape_detect_params = shape_model.predict(source=[image], show=True)
    gender_detect_params = gender_model.predict(source=[image], show=True)

    results = []

    for shape_box in shape_detect_params[0].boxes:
        s_clsID = shape_box.cls.numpy()[0]
        s_cs = shape_class_list[int(s_clsID)]

        for gender_box in gender_detect_params[0].boxes:
            g_clsID = gender_box.cls.numpy()[0]
            g_cs = gender_class_list[int(g_clsID)]

            results.append({"gender": g_cs, "shape": s_cs})

    if len(results) == 0:
        return jsonify({"error": "Image does not contain any detected objects. Kindly select a clearer image for processing"}), 400

    return jsonify(results)

@app.route('/video_detection', methods=['POST'])
def perform_video_detection():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"})

    file = request.files['file']
    video_path = "video.mp4"
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video file"})

    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        shape_detect_params = shape_model.predict(source=[frame], show=True)
        gender_detect_params = gender_model.predict(source=[frame], show=True)

        for shape_box in shape_detect_params[0].boxes:
            s_clsID = shape_box.cls.numpy()[0]
            s_cs = shape_class_list[int(s_clsID)]

            for gender_box in gender_detect_params[0].boxes:
                g_clsID = gender_box.cls.numpy()[0]
                g_cs = gender_class_list[int(g_clsID)]

                results.append({"gender": g_cs, "shape": s_cs})

    if len(results) == 0:
        return jsonify({"error": "Video does not contain any detected objects. Kindly select a clearer video for processing"}), 400

    return jsonify(results)

if __name__ == '__main__':
    app.run()
