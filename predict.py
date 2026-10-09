import json
import os
import numpy as np

# Set TensorFlow CPU thread limits before importing TF to prevent memory/CPU thrashing on free-tier cloud hosts
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# --- Model Loading ---
_model = None
IMG_SIZE = (224, 224)


def get_model():
    """Load and return the trained brain tumor model (cached)."""
    global _model, IMG_SIZE
    if _model is None:
        if os.path.exists("models/best_brain_tumor_model.h5"):
            model_path = "models/best_brain_tumor_model.h5"
        else:
            model_path = "models/brain_tumor_model.h5"
        
        print(f"Loading model from {model_path}...")
        _model = load_model(model_path)
        input_shape = _model.input_shape
        if input_shape and len(input_shape) >= 3 and input_shape[1] is not None:
            IMG_SIZE = (input_shape[1], input_shape[2])
    return _model


# Pre-warm model on module import
try:
    model = get_model()
except Exception as err:
    print(f"Warning: Failed to pre-load model: {err}")
    model = None

# --- Class Labels ---
CLASS_LABELS = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

INTERNAL_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")

# Load class indices from saved JSON if available
if os.path.exists("models/class_indices.json"):
    with open("models/class_indices.json", "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    # Build index-to-class mapping: {0: 'Glioma', 1: 'Meningioma', ...}
    display_names = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "pituitary": "Pituitary"
    }
    index_to_class = {
        class_indices.get(key, value): display_names.get(key, key.title())
        for key, value in zip(INTERNAL_CLASSES, (0, 1, 2, 3))
    }
else:
    index_to_class = CLASS_LABELS


def predict_tumor(image_path):
    """
    Predict the type of brain tumor from an MRI image.

    Args:
        image_path (str): Path to the MRI image file.

    Returns:
        dict: Prediction result containing:
            - 'label': Predicted class name (e.g., 'Glioma')
            - 'confidence': Confidence percentage (0-100)
            - 'probabilities': Dict of all class probabilities
            - 'predicted_class_index': Index of predicted class
    """
    mdl = get_model()

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = mdl.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index]) * 100

    # Build probabilities dict
    probabilities = {}
    for idx, prob in enumerate(predictions[0]):
        class_name = index_to_class.get(idx, f"Class {idx}")
        probabilities[class_name] = round(float(prob) * 100, 2)

    # Get predicted label
    label = index_to_class.get(predicted_index, f"Class {predicted_index}")

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "predicted_class_index": int(predicted_index)
    }
