# isl-alphabet-classifier

Code for training a model that can recognise ISL letters.

## Flask Web App

A simple Flask server (app.py) allows uploading an image of a hand sign and returns the predicted ISL alphabet letter with confidence using the trained model at models/isl_model.h5.

### Quick start

1) Install deps (create/activate a venv if desired):
   pip install -r requirements.txt  # if present
   pip install flask tensorflow opencv-python numpy

2) Ensure your trained model exists at models/isl_model.h5

3) Run the server:
   python app.py

4) Open in browser:
   http://localhost:5000

5) Upload an image and view the prediction.

Notes:
- Update IMG_SIZE and CLASS_NAMES in app.py if they differ from your model.
- For production, serve with a WSGI server (e.g., gunicorn) and disable debug.
