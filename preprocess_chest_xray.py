import numpy as np
from PIL import Image
import tensorflow as tf
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single chest X-ray image for model prediction.
    Returns a TensorFlow tensor of shape (1, 224, 224, 3)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)

    # Convert grayscale to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = img.resize(target_size)

    # Convert to NumPy and normalize
    img_array = np.array(img).astype('float32') / 255.0

    # Add batch dimension
    img_tensor = np.expand_dims(img_array, axis=0)

    return tf.convert_to_tensor(img_tensor)
