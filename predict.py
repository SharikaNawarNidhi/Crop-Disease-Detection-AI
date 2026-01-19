import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Model path
MODEL_PATH = 'model/crop_disease_model.h5'

# Load the trained model
model = load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = ['diseased', 'healthy']

def preprocess_image(image_path):
    """Preprocess the input image for the model."""
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image to match the model's input size
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values
    img = img / 255.0
    
    # Expand dimensions to match model input shape
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_disease(image_path):
    """Predict crop disease from an image."""
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    
    # Get the predicted class
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    
    # Get the prediction probability
    confidence = np.max(predictions[0]) * 100
    
    return predicted_class, confidence

def main():
    """Main function to run the prediction."""
    # Get image path from user
    image_path = input("Enter the path to the image: ")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print("Error: File not found.")
        return
    
    # Make prediction
    predicted_class, confidence = predict_disease(image_path)
    
    # Display results
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()