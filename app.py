import os
import logging
import json
import threading
import queue
import cv2
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from datetime import datetime

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

MODEL_URL = "https://github.com/Deepakchandrasekar05/cctv_realg_backend/releases/download/Model/best.onnx"
MODEL_PATH = "/tmp/model.onnx"

SAFETY_THRESHOLD = 0.65
violation_history = []

model_session = None
model_input_shape = (640, 640)

def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info("Downloading ONNX model from GitHub Releases...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        logging.info("Model downloaded successfully.")

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def load_model():
    global model_session
    download_model()
    providers = ['CPUExecutionProvider']
    logging.info(f"Loading {MODEL_PATH} for ONNX Runtime inference...")
    model_session = ort.InferenceSession(MODEL_PATH, providers=providers)
    logging.info("Model loaded successfully.")

@app.route('/status')
def status():
    return jsonify({"ready": model_session is not None})

@app.route('/violations')
def get_violations():
    return jsonify({
        "total_violations": len(violation_history),
        "violations": violation_history[-50:]
    })

@app.route('/detect', methods=['POST'])
def detect():
    if model_session is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = run_inference(frame)
    current_violations = [det for det in results if det['class'].startswith('NO-') and det['confidence'] > SAFETY_THRESHOLD]

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
        'detections': results,
        'violation_count': len(current_violations)
    })

def run_inference(img):
    img_resized, _, _ = letterbox(img, new_shape=model_input_shape)
    img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[np.newaxis, ...]

    inputs = {model_session.get_inputs()[0].name: img_input}
    outputs = model_session.run(None, inputs)[0]

    detections = []
    for det in outputs[0]:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        if conf > 0.4:
            detections.append({
                "class": get_class_name(cls_id),
                "confidence": float(conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
    return detections

# Map class IDs to names – update based on your training
def get_class_name(cls_id):
    class_names = {
        0: "Hardhat",
        1: "NO-Hardhat",
        2: "Mask",
        3: "NO-Mask",
        4: "Safety Vest",
        5: "NO-Safety Vest"
    }
    return class_names.get(int(cls_id), f"Class-{int(cls_id)}")

if __name__ == '__main__':
    threading.Thread(target=load_model, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)
