from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

MODEL_PATH = "yolov8s.pt"

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    status = {"server": "running", "model_loaded": model is not None}
    return jsonify(status), 200

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file part. Please send multipart/form-data with field 'image'."}), 400

        file = request.files['image']
        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        if model is None:
            return jsonify({"detections": [], "filename": filename, "annotated_image": None}), 200

        results = model.predict(save_path, imgsz=640, conf=0.2)

        detections = []
        img = cv2.imread(save_path)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist() 

                detections.append({"class": cls_name, "confidence": conf, "bbox": bbox})

                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, img)

        return jsonify({
            "detections": detections,
            "filename": filename,
            "annotated_image": f"/outputs/{filename}"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
