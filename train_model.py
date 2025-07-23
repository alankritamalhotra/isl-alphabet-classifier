import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

DATA_DIR = "isl-dataset"  # rename this if you named your folder something else
IMG_SIZE = 64

# Load images and labels
def load_data():
    X, y = [], []
    labels = sorted(os.listdir(DATA_DIR))
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        path = os.path.join(DATA_DIR, label)
        for img in tqdm(os.listdir(path), desc=f"Loading {label}"):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                X.append(img_array)
                y.append(label_map[label])
            except:
                continue

    return np.array(X), np.array(y), label_map

# Preprocess
X, y, label_map = load_data()
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
if not os.path.exists("models"):
    os.mkdir("models")
model.save("models/isl_model.h5")
