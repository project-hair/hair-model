from flask import Flask, jsonify
from ultralytics import YOLO
import cv2
import numpy as np


class ImageDetector:
    def __init__(self):
        self.shape_model = YOLO("./utils/yolo8n-shapes.pt", "v8")
        self.gender_model = YOLO("./utils/yolov8-gender.pt", "v8")
        self.shape_class_list = self.read_class_data("utils/coco.txt")
        self.gender_class_list = self.read_class_data("utils/gender.txt")

    def read_class_data(self, file_path):
        with open(file_path, "r") as file:
            class_list = file.read().splitlines()
        return class_list

    def detect_objects(self, image):
        shape_detect_params = self.shape_model.predict(source=[image])
        gender_detect_params = self.gender_model.predict(source=[image])


        for shape_box in shape_detect_params[0].boxes:
            s_clsID = shape_box.cls.numpy()[0]
            s_cs = self.shape_class_list[int(s_clsID)]

            for gender_box in gender_detect_params[0].boxes:
                g_clsID = gender_box.cls.numpy()[0]
                g_cs = self.gender_class_list[int(g_clsID)]

                results = {"gender": g_cs, "shape": s_cs}

        return results

    def perform_detection(self):
        if 'file' not in self.request.files:
            return jsonify({"error": "No file found"})

        file = self.request.files['file']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        results = image_detector.detect_objects(image)

        if len(results) == 0:
            return jsonify({"error": "Image does not contain any detected objects. Kindly select a clearer image for processing"}), 400

        return jsonify(results)
image_detector = ImageDetector()
