import logging
import threading
import queue
import json
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import gdown
import onnxruntime as ort

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
socketio = SocketIO(app, cors_allowed_origins="*")

frame_queue = queue.Queue(maxsize=10)
latest_detections = []
detection_event = threading.Event()
violation_history = []
SAFETY_THRESHOLD = 0.65

model_path = "best.onnx"
drive_file_id = "1JWVH0gQ4e6KVJERCNyQRgnD8FNP3pOZc"
gdown_url = f"https://drive.google.com/uc?id={drive_file_id}"

# 🔽 Download model if not exists
if not os.path.exists(model_path):
    print("🔄 Downloading ONNX model from Google Drive...")
    downloaded = gdown.download(gdown_url, model_path, quiet=False)
    if downloaded is None or not os.path.exists(model_path):
        raise FileNotFoundError("❌ Failed to download the ONNX model.")
    print("✅ Model downloaded successfully.")

# 🔁 Load ONNX model using ONNX Runtime
try:
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    model_loaded = True
    print("✅ ONNX model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")
    model_loaded = False

@app.route('/')
def home():
    return "🚀 CCTV ONNX Flask App is running!"

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
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        roi_info = json.loads(request.form['roi']) if 'roi' in request.form else None
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
                "image": cv2.imencode('.jpg', frame)[1].tobytes().hex()
            }
            violation_history.append(violation_data)
            socketio.emit('new_violation', violation_data)

        return jsonify({
            'detections': detections,
            'violation_count': len(current_violations)
        })

    except Exception as e:
        logging.error(f"Error during detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_frame(frame, roi_info=None):
    try:
        if roi_info and 'offset' in roi_info:
            x = int(roi_info['offset']['x'])
            y = int(roi_info['offset']['y'])
            width = int(roi_info['width'])
            height = int(roi_info['height'])
            h, w = frame.shape[:2]

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = max(1, min(width, w - x))
            height = max(1, min(height, h - y))
            frame = frame[y:y + height, x:x + width]

        # Preprocess
        resized = cv2.resize(frame, (640, 640))
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True, crop=False)
        blob = blob.astype(np.float32)

        # Inference
        outputs = session.run(None, {input_name: blob})[0]

        # Postprocess (⚠️ requires adapting to your model's output format)
        detections = []
        for det in outputs:
            conf = float(det[4].item()) if isinstance(det[4], np.ndarray) else float(det[4])
            class_id = int(det[5].item()) if isinstance(det[5], np.ndarray) else int(det[5])
            if conf > SAFETY_THRESHOLD:
                x1, y1, x2, y2 = [int(x.item()) if isinstance(x, np.ndarray) else int(x) for x in det[:4]]
                detections.append({
                    'class': f'class_{class_id}',
                    'confidence': round(conf, 2),
                    'bbox': [x1, y1, x2, y2]
                })

        return detections

    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        return []

@app.route('/test-cors')
def test_cors():
    return jsonify({"message": "CORS is working!"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
