import logging
import threading
import queue
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
from flask_socketio import SocketIO, emit

logging.basicConfig(filename='web.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

frame_queue = queue.Queue(maxsize=10)
latest_detections = []
detection_event = threading.Event()

def load_model():
    global model
    model = YOLO('H:/Construction_Safety/project/backend/best.onnx', task='detect')
    _ = model(np.zeros((640, 480, 3), dtype=np.uint8))
    logging.info("Model loaded and warmed up")

threading.Thread(target=load_model, daemon=True).start()

@app.route('/status')
def status():
    return jsonify({"ready": "model" in globals()})

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
        return jsonify({'detections': detections})
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
            # Ensure cropping is within bounds
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            width = max(1, min(width, w-x))
            height = max(1, min(height, h-y))
            frame = frame[y:y+height, x:x+width]
            logging.info(f"Processing cropped frame at offset ({x},{y}) size ({width}x{height})")
        else:
            logging.info("Processing full frame")
        results = model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if conf > 0.5:
                    detections.append({
                        'class': class_name,
                        'confidence': round(conf, 2),
                        'bbox': [x1, y1, x2, y2]
                    })
        person_detections = [d for d in detections if d['class'] == 'Person']
        return person_detections
    except Exception as e:
        logging.error(f"Cropping or detection error: {str(e)}")
        return []

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)