import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load model
model = tf.keras.models.load_model('../models/cat_dog_model.keras')

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] > 0.8:
        return "Dog"
    elif prediction[0] < 0.2:
        return "Cat"
    else:
        return "Unknown"





# Example usage
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
else:
    image_path = sys.argv[1]
    result = predict_image(image_path)
    print(f"The image is classified as: {result}")
