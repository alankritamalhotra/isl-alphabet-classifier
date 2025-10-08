import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = keras.models.load_model('isl_cnn_model.h5')

# ISL alphabet classes (A-Z)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Path to the test images directory (Mendeley dataset)
test_data_path = 'test_images'

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    return img_array

def evaluate_test_images():
    """Evaluate the model on test images and generate metrics"""
    predictions = []
    true_labels = []
    
    print("\n" + "="*60)
    print("ISL ALPHABET CLASSIFIER - REAL-WORLD IMAGE EVALUATION")
    print("="*60 + "\n")
    
    # Iterate through each class folder
    for class_name in classes:
        class_path = os.path.join(test_data_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: Directory not found for class {class_name}")
            continue
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Preprocess and predict
                img_array = load_and_preprocess_image(img_path)
                img_batch = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_batch, verbose=0)
                predicted_class = classes[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                predictions.append(predicted_class)
                true_labels.append(class_name)
                
                print(f"Image: {img_file:30s} | True: {class_name} | "
                      f"Predicted: {predicted_class} | Confidence: {confidence:.2f}%")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    if len(predictions) == 0:
        print("\nError: No test images found. Please ensure test images are in the 'test_images' directory.")
        return
    
    # Calculate per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60 + "\n")
    
    for class_name in classes:
        class_indices = [i for i, label in enumerate(true_labels) if label == class_name]
        if len(class_indices) > 0:
            correct = sum([1 for i in class_indices if predictions[i] == class_name])
            accuracy = (correct / len(class_indices)) * 100
            print(f"Class {class_name}: {accuracy:.2f}% ({correct}/{len(class_indices)} correct)")
    
    # Overall accuracy
    overall_accuracy = (sum([1 for i in range(len(predictions)) 
                            if predictions[i] == true_labels[i]]) / len(predictions)) * 100
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60 + "\n")
    print(classification_report(true_labels, predictions, target_names=classes, zero_division=0))
    
    # Confusion matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60 + "\n")
    
    cm = confusion_matrix(true_labels, predictions, labels=classes)
    
    # Print confusion matrix in text format
    print("     ", end="")
    for c in classes:
        print(f"{c:>4}", end="")
    print()
    
    for i, c in enumerate(classes):
        print(f"{c:>4} ", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>4}", end="")
        print()
    
    # Save confusion matrix as image
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - ISL Alphabet Classifier\n(Real-World Mendeley Test Images)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    if not os.path.exists('isl_cnn_model.h5'):
        print("Error: Model file 'isl_cnn_model.h5' not found.")
        print("Please train the model first using train_model.py")
    elif not os.path.exists(test_data_path):
        print(f"Error: Test data directory '{test_data_path}' not found.")
        print("Please download and extract the Mendeley ISL dataset to the 'test_images' folder.")
    else:
        evaluate_test_images()
