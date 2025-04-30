from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import gdown

app = Flask(__name__)
CORS(app)

# ✅ Correct Google Drive file ID-based download URL
model_path = "best.onnx"
drive_file_id = "1JWVH0gQ4e6KVJERCNyQRgnD8FNP3pOZc"
gdown_url = f"https://drive.google.com/uc?id={drive_file_id}"

# 🔄 Download ONNX model if not present
if not os.path.exists(model_path):
    print("🔄 Downloading ONNX model from Google Drive...")
    gdown.download(gdown_url, model_path, quiet=False)
    print("✅ Model downloaded successfully.")

# ✅ Load YOLO ONNX model
try:
    model = YOLO(model_path)
    model_loaded = True
    print("✅ Model loaded and ready.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model_loaded = False

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

    try:
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

    except Exception as e:
        print(f"❌ Inference error: {e}")
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
