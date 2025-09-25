from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLO model
MODEL_PATH = "D:\\VIT AIDS\\Module V\\EDI\\Real_time_object_detection_yolo\\runs\\detect\\final_best_train4\\weights\\best.pt"

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return "Server is running", 200

@app.route("/health", methods=["GET"])
def health():
    """
    Health-check endpoint to verify server and model status.
    """
    status = {
        "server": "running",
        "model_loaded": model is not None
    }
    return jsonify(status), 200

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No file part. Please send multipart/form-data with field 'image'."}), 400

    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # If model not loaded, return dummy
    if model is None:
        return jsonify({"detections": "dummy", "filename": filename}), 200

    # Run YOLO model prediction
    results = model.predict(save_path, imgsz=640, conf=0.25)

    # Parse detections into JSON
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()  
            })

    return jsonify({"detections": detections, "filename": filename}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
