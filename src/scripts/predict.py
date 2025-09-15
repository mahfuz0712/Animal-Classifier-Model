import tensorflow as tf
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np
import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load model
model = tf.keras.models.load_model('../models/animal_model.keras')

def predict_image(image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the trained model
        prediction = model.predict(img_array)

        # Assuming the model has 4 classes (Dog, Cat, Cow, Unknown)
        class_names = ["Dog", "Cat", "Cow", "Unknown"]  # Class labels in the order of softmax output

        # Get the predicted class index (the one with the highest probability)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Return the predicted class name
        return class_names[predicted_class_index]
    except Exception as e:
        return str(e)

# Example usage
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
else:
    image_path = sys.argv[1]
    result = predict_image(image_path)
    print(f"The image is classified as: {result}")