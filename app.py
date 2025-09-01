import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import wfdb
import neurokit2 as nk
from pathlib import Path
from PIL import Image
import io
import base64
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- Configuration ---
UPLOAD_FOLDER = Path.home() / 'ecg_uploads'
ECG_MODEL_PATH = Path.home() / 'best_ecg_model.keras'
SAMPLING_RATE = 250
WINDOW_SAMPLES = 10 * SAMPLING_RATE
ALLOWED_ECG = {'dat', 'hea'}
ALLOWED_XRAY = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# --- Models ---
ecg_model = None
xray_model = None
xray_processor = None

def load_models():
    global ecg_model, xray_model, xray_processor
    print("Loading ECG model...")
    ecg_model = tf.keras.models.load_model(ECG_MODEL_PATH)
    print("ECG model loaded.")

    print("Loading X-ray ViT model...")
    repo_id = "codewithdark/vit-chest-xray"
    xray_processor = AutoImageProcessor.from_pretrained(repo_id)
    xray_model = AutoModelForImageClassification.from_pretrained(repo_id)
    xray_model.eval()
    print("X-ray model loaded.")

# --- Helpers ---
def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext

def preprocess_ecg(dat_path, hea_path):
    try:
        record_name = Path(dat_path).stem
        record = wfdb.rdrecord(str(Path(dat_path).parent / record_name), sampto=WINDOW_SAMPLES)
        if record.fs != SAMPLING_RATE: return None
        signal = record.p_signal[:, 0]
        if len(signal) < WINDOW_SAMPLES: return None
        cleaned = nk.ecg_clean(signal, sampling_rate=SAMPLING_RATE)
        return cleaned.reshape((1, WINDOW_SAMPLES, 1)).astype(np.float32)
    except Exception as e:
        print(f"ECG Preprocessing Error: {e}")
        return None

def preprocess_xray(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inputs = xray_processor(images=img, return_tensors="pt")
        return inputs
    except Exception as e:
        print(f"X-ray Preprocessing Error: {e}")
        return None

# --- Routes ---
@app.route('/predict_ecg', methods=['POST'])
def predict_ecg():
    if 'dat_file' not in request.files or 'hea_file' not in request.files:
        return jsonify({'error': 'Both .dat and .hea files are required'}), 400

    dat_file = request.files['dat_file']
    hea_file = request.files['hea_file']

    if not (allowed_file(dat_file.filename, {'dat'}) and allowed_file(hea_file.filename, {'hea'})):
        return jsonify({'error': 'Invalid ECG file types'}), 400

    dat_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(dat_file.filename))
    hea_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(hea_file.filename))
    dat_file.save(dat_path)
    hea_file.save(hea_path)

    try:
        tensor = preprocess_ecg(dat_path, hea_path)
        if tensor is None:
            return jsonify({'error': 'Could not process ECG files'}), 500

        score = float(ecg_model.predict(tensor)[0][0])
        result = {
            'prediction': 'CHF Positive' if score > 0.5 else 'CHF Negative',
            'confidence_score': score
        }
        return jsonify(result)
    finally:
        if os.path.exists(dat_path): os.remove(dat_path)
        if os.path.exists(hea_path): os.remove(hea_path)

@app.route('/predict_xray', methods=['POST'])
def predict_xray():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_bytes = file.read()

    if not allowed_file(file.filename, ALLOWED_XRAY):
        return jsonify({'error': 'Invalid image file type'}), 400

    inputs = preprocess_xray(image_bytes)
    if inputs is None:
        return jsonify({'error': 'Could not process image'}), 500

    with torch.no_grad():
        outputs = xray_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)

    label = xray_model.config.id2label[class_idx.item()]
    result = {
        'prediction': label,
        'confidence_score': float(confidence.item()),
        'gradcam_image_base64': None  # Can implement later for explainability
    }

    return jsonify(result)

# --- Run App ---
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    load_models()
    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
