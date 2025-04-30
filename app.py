import logging
import threading
import queue
import json
import cv2
import numpy as np
import os
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
from flask_socketio import SocketIO, emit

logging.basicConfig(filename='web.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

frame_queue = queue.Queue(maxsize=10)
latest_detections = []
detection_event = threading.Event()
violation_history = []
SAFETY_THRESHOLD = 0.65  # Higher confidence threshold for safety equipment

# ✅ Replace this with your direct model download link (GitHub Release / HuggingFace)
MODEL_URL = "https://github.com/Deepakchandrasekar05/cctv_realg_backend/releases/download/Model/best.onnx"
MODEL_PATH = "/tmp/model.onnx"

def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info("Download complete.")
        else:
            logging.error(f"Failed to download model: {response.status_code}")
            raise RuntimeError("Model download failed")

def load_model():
    global model
    download_model()
    model = YOLO(MODEL_PATH, task='detect')
    #_ = model(np.zeros((640, 480, 3), dtype=np.uint8))  # Warm up
    logging.info("Model loaded and warmed up.")

# Run model loading in background thread
threading.Thread(target=load_model, daemon=True).start()

@app.route('/status')
def status():
    return jsonify({"ready": "model" in globals()})

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
        logging.error(f"Error during object detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_frame(frame, roi_info=None):
    if 'model' not in globals():
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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
