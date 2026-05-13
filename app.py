import os
import cv2
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO models
MODELS = {
    'yolov9':  'best_y9.pt' if os.path.exists('best_y9.pt') else 'results/best_y9.pt',
    'yolov10': 'best_y10.pt' if os.path.exists('best_y10.pt') else 'results/best_y10.pt',
    'yolov11': 'best_y11.pt' if os.path.exists('best_y11.pt') else 'results/best_y11.pt'
}

# Fallbacks to pre-trained weights if fine-tuned ones aren't found
loaded_models = {}
for version, path in MODELS.items():
    # Fix: YOLOv11 fallback is 'yolo11n.pt', others are 'yolov9c.pt' etc.
    fallback = f"{version if 'v11' not in version else 'yolo11'}n.pt" if 'v11' in version else f"{version}{'c' if 'v9' in version else 's'}.pt"
    actual_path = path if os.path.exists(path) else fallback
    print(f"Loading {version} from: {actual_path}")
    try:
        loaded_models[version] = YOLO(actual_path)
    except Exception as e:
        print(f"Failed to load {version}: {e}")

print("Models loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get model version from request, default to yolov11
    model_version = request.form.get('version', 'yolov11')
    if model_version not in loaded_models:
        return jsonify({'error': f'Model version {model_version} not loaded'}), 400
    
    model = loaded_models[model_version]

    # Save uploaded file with unique name
    unique_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(secure_filename(file.filename))[1] or '.jpg'
    filename = f"{unique_id}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Run inference - reset to 0.20 for a good balance of precision/recall
        # The new SeaClear model is much more accurate and handles noise better
        # Lowering to 0.05 for debugging to see everything the model finds in the logs
        results = model.predict(source=filepath, conf=0.05, imgsz=320, save=False)

        print(f"DEBUG: Found {len(results[0].boxes)} raw detections at 0.05 conf")
        for box in results[0].boxes:
            print(f"DEBUG: Class {int(box.cls)} ({model.names[int(box.cls)]}) - Conf: {float(box.conf):.3f}")

        if not results:
            return jsonify({'error': 'No results from model'}), 500

        # Process detections and annotate image
        detections = []
        
        # Process the first result (single image)
        r = results[0]
        orig_h, orig_w = r.orig_shape
        
        # Load the original image for manual annotation
        annotated_img = cv2.imread(filepath)

        # Class mapping for display
        class_map = {
            'plastic-bag': 'trash_Plastic',
            'waste': 'Object',
            'bottle': 'Object',
            'can': 'Object'
        }

        for box in r.boxes:
            # Setting to 0.05 so the user can see all detected barrels in the demo
            if float(box.conf) < 0.05:
                continue

            # Get pixel coordinates and ensure they are within image bounds
            raw_x1, raw_y1, raw_x2, raw_y2 = [float(x) for x in box.xyxy[0]]
            
            x1 = max(0, int(raw_x1))
            y1 = max(0, int(raw_y1))
            x2 = min(orig_w, int(raw_x2))
            y2 = min(orig_h, int(raw_y2))
            
            # Calculate width and height in pixels
            w = x2 - x1
            h = y2 - y1
            
            # Map class names to user preference
            raw_class = model.names[int(box.cls)]
            mapped_class = class_map.get(raw_class, raw_class)
            conf = round(float(box.conf), 3)

            # Log localization for debugging
            print(f"DEBUG: Detected {mapped_class} at [{x1}, {y1}, {x2}, {y2}] with conf {conf}")

            # Draw Yellow Box (BGR: 0, 255, 255)
            color = (0, 255, 255) # Yellow
            thickness = 2
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)

            # Draw Label Background
            label = f"{mapped_class} {conf}"
            (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            # Ensure label doesn't go off top of image
            text_y = max(y1, 20)
            cv2.rectangle(annotated_img, (x1, text_y - 20), (x1 + t_w, text_y), color, -1)
            
            # Draw Label Text (Black)
            cv2.putText(annotated_img, label, (x1, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Add to metadata
            detections.append({
                'class': mapped_class,
                'confidence': conf,
                'bbox': [float(x) for x in [x1, y1, x2, y2]],
                'dimensions': {'width': float(w), 'height': float(h)}
            })

        # Encode to JPEG bytes then base64
        success, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = None
        if success:
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'image_size': {'width': orig_w, 'height': orig_h},
            'result_image_b64': annotated_b64
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'loaded_models': list(loaded_models.keys()),
        'trained_weights': MODELS
    })

if __name__ == '__main__':
    # Cloud environments like Hugging Face use port 7860
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
