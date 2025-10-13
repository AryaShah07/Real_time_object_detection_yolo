from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
from flask_cors import CORS 
from flask import Response
import time
import pandas as pd
import base64
from PIL import Image
import io
import numpy as np


app = Flask(__name__, static_url_path='/static', static_folder='static') 

CORS(app)


MODEL_PATH = "yolov8s.pt"

try:
    model = YOLO(MODEL_PATH)
    print(f" Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/style.css")
def serve_style_css():
    return send_from_directory(app.static_folder, 'style.css')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/about.html", methods=["GET"])
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route("/health", methods=["GET"])
def health():
    status = {"server": "running", "model_loaded": model is not None}
    return jsonify(status), 200

@app.route("/results.html")
def results_partial():
    return render_template("results.html")

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
            return jsonify({"detections": [], "filename": filename, "annotated_image": None, "error": "Object detection model not loaded on the server."}), 500

        confidence_threshold = float(request.form.get('confidence', 0.2))
        results = model.predict(save_path, imgsz=640, conf=confidence_threshold,verbose=False)


        detections = []
        img = cv2.imread(save_path)
        if img is None:
            return jsonify({"error": f"Could not read the uploaded image: {filename}. It might be corrupted or not a valid image format."}), 400

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

        output_filename = "annotated_" + filename
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, img)
        annotated_image_url = f"/outputs/{output_filename}"

        
        csv_url = None
        if detections:
            df = pd.DataFrame(detections)
            csv_filename = "predictions_" + filename.rsplit('.', 1)[0] + ".csv"
            csv_path = os.path.join(OUTPUT_FOLDER, csv_filename)
            df.to_csv(csv_path, index=False)
            csv_url = f"/outputs/{csv_filename}"

        return jsonify({
            "detections": detections,
            "filename": filename,
            "annotated_image": annotated_image_url,
            "csv_url": csv_url
        }), 200
    
        

    except Exception as e:
        return jsonify({"error": str(e)}), 500





# Live webcam endpoint
def generate_frames():
    """Continuously capture webcam frames and stream YOLO detections as MJPEG."""
    camera = cv2.VideoCapture(0) 
    if not camera.isOpened():
        raise RuntimeError("❌ Could not open webcam. Check camera permissions.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        
        results = model.predict(frame, imgsz=640, conf=0.3)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()

                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

       
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        
        time.sleep(0.03)

    camera.release()


@app.route("/live")
def live():
    """MJPEG stream for real-time webcam detection."""
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detect_live_frame", methods=["POST"])
def detect_live_frame():
    """Fast endpoint for live frame detection from Streamlit"""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data"}), 400
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get parameters
        frame_data = data['frame']
        confidence = data.get('confidence', 0.4)
        
        # Convert base64 to image
        import base64
        from PIL import Image
        import io
        import numpy as np
        
        image_data = base64.b64decode(frame_data.split(',')[1])
        image_pil = Image.open(io.BytesIO(image_data))
        image_np = np.array(image_pil)
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run YOLO detection (faster, no tracking for live)
        results = model.predict(frame, imgsz=640, conf=confidence, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    if conf >= confidence:
                        detections.append({
                            "class": cls_name,
                            "class_id": cls_id,
                            "confidence": conf,
                            "bbox": [int(x) for x in bbox]
                        })
        
        return jsonify({
            "detections": detections,
            "count": len(detections),
            "frame_processed": True
        })
        
    except Exception as e:
        print(f"Live detection error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8000, debug=True)

