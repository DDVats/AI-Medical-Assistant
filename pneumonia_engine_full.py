import os
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import streamlit as st  # For caching

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image from file, converts to RGB, resizes, normalizes, and adds batch dimension.
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    return tf.convert_to_tensor(img_tensor)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_pneumonia_model():
    """
    Downloads and loads the correct pneumonia detection model from Hugging Face Hub.
    """
    repo_id = "ayushirathour/chest-xray-pneumonia-detection"
    filename = "best_chest_xray_model.h5"  # Correct model file

    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# ---------------------------
# Inference Function
# ---------------------------
def predict_pneumonia(model, image_path, threshold=0.5):
    """
    Predicts Pneumonia or Normal for a single image.
    Returns a dictionary with label and confidence.
    """
    img_tensor = preprocess_image(image_path)
    prob = model.predict(img_tensor)[0][0]
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    return {"label": label, "confidence": float(prob)}

# ---------------------------
# Batch Prediction
# ---------------------------
def predict_on_folder(folder_path):
    """
    Loops through all images in a folder and subfolders and prints predictions.
    """
    model = load_pneumonia_model()
    if model is None:
        print("Model failed to load.")
        return

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, file)
                result = predict_pneumonia(model, img_path)
                print(f"{img_path}: Prediction={result['label']}, Confidence={result['confidence']:.2f}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Example: run predictions on the train folder
    folder_to_test = "/home/vatsdd/Downloads/chest-xray-pneumonia/chest_xray/train"
    predict_on_folder(folder_to_test)
