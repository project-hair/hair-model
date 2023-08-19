from flask import Flask, jsonify, request
from video_detect import VideoDetector
from image_detect import ImageDetector
from db_config import db_config
import mysql.connector
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
        return jsonify({"error": "Image does not contain any detected objects. Kindly select a clearer image for processing"}), 500

    return jsonify(results)



@app.route('/video_detection', methods=['GET'])
def perform_video_detection():
    return video_detector.process_frame()



@app.route('/api/all-products/', methods=['GET'])
def get_data():
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()

    query = 'SELECT * FROM hair'
    cursor.execute(query)

    data = cursor.fetchall()
    cursor.close()
    cnx.close()
    result = []

    for row in data:
        result.append({
            'hairId': row[0],
            'gender': row[1],
            'shape': row[2],
            'url': row[3],
        })
    return jsonify(result)



@app.route('/api/filter/', methods=['POST'])
def filter():
    data = request.json
    gender = data.get('gender')
    shape = data.get('shape')

    if not gender or not shape:
        return jsonify({'error': 'Both gender and shape parameters are required.'}), 400

    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()
    query = 'SELECT * FROM hair WHERE hair.gender = %s AND hair.shape = %s'
    cursor.execute(query, (gender, shape))
    data = cursor.fetchall()

    cursor.close()
    cnx.close()
    result = []

    for row in data:
        result.append({
            'hairId': row[0],
            'gender': row[1],
            'shape': row[2],
            'url': row[3],
        })
    return jsonify(result)




if __name__ == '__main__':
    app.run()
