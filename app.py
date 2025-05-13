import base64
import logging
import threading
import queue
import json
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import os
import gdown

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://realg-c55af.web.app"])
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5173", "https://realg-c55af.web.app"], async_mode='threading')

SAFETY_THRESHOLD = 0.65
violation_history = []

# ✅ Load ONNX model dynamically
model_path = "best.onnx"
drive_file_id = "1JWVH0gQ4e6KVJERCNyQRgnD8FNP3pOZc"
gdown_url = f"https://drive.google.com/uc?id={drive_file_id}"

def load_model():
    global model
    if not os.path.exists(model_path):
        logging.info("Downloading model...")
        gdown.download(gdown_url, model_path, quiet=False)
    model = YOLO(model_path, task="detect")
    _ = model(np.zeros((640, 480, 3), dtype=np.uint8))
    logging.info("Model loaded.")

threading.Thread(target=load_model, daemon=True).start()

# 🔁 WebSocket handler
@socketio.on('frame')
def handle_frame(data):
    try:
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        roi_info = data.get('roi', None)
        detections = process_frame(frame, roi_info)

        current_violations = [
            d for d in detections
            if d['class'] in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'] and d['confidence'] > SAFETY_THRESHOLD
        ]

        if current_violations:
            timestamp = datetime.now().isoformat()
            violation_data = {
                "timestamp": timestamp,
                "violations": current_violations,
                "image": base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
            }
            violation_history.append(violation_data)
            emit('violation', violation_data)

        emit('detection_result', {
            'detections': detections,
            'violation_count': len(current_violations)
        })

    except Exception as e:
        emit('detection_result', {'error': str(e)})
        logging.error(f"Frame processing error: {e}")

def process_frame(frame, roi_info=None):
    try:
        if roi_info and 'offset' in roi_info:
            x, y = int(roi_info['offset']['x']), int(roi_info['offset']['y'])
            width, height = int(roi_info['width']), int(roi_info['height'])
            h, w = frame.shape[:2]
            x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
            width, height = max(1, min(width, w - x)), max(1, min(height, h - y))
            frame = frame[y:y + height, x:x + width]

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
        logging.error(f"process_frame error: {e}")
        return []

@app.route('/')
def home():
    return "WebSocket Detection Server Running!"

@app.route('/violations')
def get_violations():
    return jsonify({
        "total_violations": len(violation_history),
        "violations": violation_history[-50:]
    })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
