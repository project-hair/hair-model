from flask import Flask, jsonify, request
from video_detect import VideoDetector
from image_detect import ImageDetector
import cv2
import numpy as np

app = Flask(__name__)

image_detector = ImageDetector()
video_detector = VideoDetector()

@app.route('/image_detection', methods=['POST'])
def perform_image_detection():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"})

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results = image_detector.detect_objects(image)

    if len(results) == 0:
        return jsonify({"error": "Image does not contain any detected objects. Kindly select a clearer image for processing"}), 400

    return jsonify(results)


@app.route('/video_detection', methods=['GET'])
def perform_video_detection():
    return video_detector.process_frame()


if __name__ == '__main__':
    app.run()
