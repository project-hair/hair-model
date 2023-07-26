from flask import Flask, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np

class VideoDetector:
    def __init__(self):
        self.shape_model = YOLO("./utils/yolo8n-shapes.pt", "v8")
        self.gender_model = YOLO("./utils/yolov8-gender.pt", "v8")
        self.shape_class_list = self.read_class_data("utils/coco.txt")
        self.gender_class_list = self.read_class_data("utils/gender.txt")
        self.cap = cv2.VideoCapture(0)

    def read_class_data(self, file_path):
        with open(file_path, "r") as file:
            class_list = file.read().splitlines()
        return class_list

    def detect_objects(self, frame):
        shape_detect_params = self.shape_model.predict(source=[frame], show=True)
        gender_detect_params = self.gender_model.predict(source=[frame], show=True)


        for shape_box in shape_detect_params[0].boxes:
            s_clsID = shape_box.cls.numpy()[0]
            s_cs = self.shape_class_list[int(s_clsID)]

            for gender_box in gender_detect_params[0].boxes:
                g_clsID = gender_box.cls.numpy()[0]
                g_cs = self.gender_class_list[int(g_clsID)]

                results = {"gender": g_cs, "shape": s_cs}

        return results

    def process_frame(self):
        if not self.cap.isOpened():
            return jsonify({"error": "Cannot open camera"})

        ret, frame = self.cap.read()
        if not ret:
            return jsonify({"error": "Failed to read frame from camera"})

        results = self.detect_objects(frame)

        if len(results) == 0:
            return jsonify({"error": "Frame does not contain any detected objects. Kindly select a clearer frame for processing"}), 400

        return jsonify(results)

video_detector = VideoDetector()
