from flask import Flask, request, render_template_string
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os

# Config
MODEL_PATH = 'models/isl_model.h5'
IMG_SIZE = (64, 64)  # adjust to your model's input size
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Load model once at startup
model = load_model(MODEL_PATH)

HTML_FORM = """
<!doctype html>
<title>ISL Letter Classifier</title>
<h1>Upload a hand sign image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=image accept="image/*">
  <input type=submit value=Predict>
</form>
{% if pred is defined %}
  <h2>Prediction: {{ pred }} ({{ (conf*100)|round(2) }}% confidence)</h2>
{% endif %}
"""

def preprocess_image(file_stream):
    # Read file bytes to NumPy array
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Invalid image file')

    # Convert to RGB if model expects; here keep BGR->RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # Normalize to [0,1]
    img = img.astype('float32') / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template_string(HTML_FORM, pred='No file provided', conf=0)
        f = request.files['image']
        try:
            x = preprocess_image(f.stream)
            preds = model.predict(x)
            if preds.ndim == 2 and preds.shape[1] == len(CLASS_NAMES):
                probs = preds[0]
            else:
                # Handle sigmoid single-output per class models if needed
                probs = preds.flatten()
                if probs.size != len(CLASS_NAMES):
                    raise ValueError('Model output shape does not match class count')
            idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[idx]
            conf = float(probs[idx])
            return render_template_string(HTML_FORM, pred=pred_class, conf=conf)
        except Exception as e:
            return render_template_string(HTML_FORM, pred=f'Error: {str(e)}', conf=0)
    return render_template_string(HTML_FORM)

if __name__ == '__main__':
    # Allow host override for container use
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
