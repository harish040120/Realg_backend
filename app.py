from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import requests

app = Flask(__name__)
CORS(app)

# Load YOLO model from Google Drive (if not already present)
model_path = "best.onnx"
model_url = "https://drive.google.com/uc?id=1mYZvpVaJWQl2PFCrjRWTPm3zRikmXgZiu&export=download"

if not os.path.exists(model_path):
    print("Downloading model...")
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    print("Model downloaded.")

# Load YOLO model
model = YOLO(model_path)
model_loaded = True  # Signal that the model is ready

@app.route('/')
def home():
    return "🚀 YOLO ONNX Flask App is running!"

@app.route('/status')
def status():
    return jsonify({"ready": model_loaded})

@app.route('/detect', methods=['POST'])
def detect():
    if not model_loaded:
        return jsonify({'error': 'Model not yet loaded'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img_np = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    results = model(img)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': round(conf, 2),
            'class': cls
        })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
