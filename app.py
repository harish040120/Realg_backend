import logging
import threading
import queue
import json
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import gdown

# ---------------- Flask + CORS Setup -------------------
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)

# ---------------- SocketIO Setup -----------------------
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------- Globals ------------------------------
frame_queue = queue.Queue(maxsize=10)
latest_detections = []
detection_event = threading.Event()
violation_history = []
SAFETY_THRESHOLD = 0.65

# ---------------- Download YOLO Model ------------------
model_path = "best.onnx"
drive_file_id = "1JWVH0gQ4e6KVJERCNyQRgnD8FNP3pOZc"
gdown_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    print("🔄 Downloading ONNX model from Google Drive...")
    gdown.download(gdown_url, model_path, quiet=False)
    print("✅ Model downloaded successfully.")

# ---------------- Load Model ---------------------------
try:
    model = YOLO(model_path)
    model_loaded = True
    print("✅ Model loaded and ready.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model_loaded = False

# ---------------- API Routes ---------------------------

@app.route('/')
def home():
    return "🚀 YOLO ONNX Flask App is running!"

@app.route('/status')
def status():
    return jsonify({"ready": model_loaded})

@app.route('/violations')
def get_violations():
    return jsonify({
        "total_violations": len(violation_history),
        "violations": violation_history[-50:]
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        roi_info = None
        if 'roi' in request.form:
            roi_info = json.loads(request.form['roi'])

        detections = process_frame(frame, roi_info)

        current_violations = []
        for d in detections:
            if d['class'] in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and d['confidence'] > SAFETY_THRESHOLD:
                current_violations.append(d)

        if current_violations:
            timestamp = datetime.now().isoformat()
            violation_data = {
                "timestamp": timestamp,
                "violations": current_violations,
                "image": cv2.imencode('.jpg', frame)[1].tobytes().hex()
            }
            violation_history.append(violation_data)
            socketio.emit('new_violation', violation_data)

        return jsonify({
            'detections': detections,
            'violation_count': len(current_violations)
        })

    except Exception as e:
        logging.error(f"Error during object detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------- Frame Processing ---------------------

def process_frame(frame, roi_info=None):
    if not model_loaded:
        return []

    try:
        if roi_info and 'offset' in roi_info:
            x = int(roi_info['offset']['x'])
            y = int(roi_info['offset']['y'])
            width = int(roi_info['width'])
            height = int(roi_info['height'])
            h, w = frame.shape[:2]

            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            width = max(1, min(width, w-x))
            height = max(1, min(height, h-y))
            frame = frame[y:y+height, x:x+width]

        results = model(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                min_conf = SAFETY_THRESHOLD if class_name == 'Person' else 0.5

                if conf > min_conf:
                    detections.append({
                        'class': class_name,
                        'confidence': round(conf, 2),
                        'bbox': [x1, y1, x2, y2]
                    })

        return detections

    except Exception as e:
        logging.error(f"Cropping or detection error: {str(e)}")
        return []

# ---------------- Run App ------------------------------

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
