import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def predict_image(model_path, image_path):
    try:
        # Load the model from the specified path
        model = tf.keras.models.load_model(model_path)
        
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the loaded model
        prediction = model.predict(img_array)

        # Assuming the model is for binary classification (your train script implies this)
        # 0 or 1, let's assume 0 = cat, 1 = dog based on the directory names
        class_names = ["Cat", "Dog"] # Adjust based on your actual data

        # Get the predicted class index (0 or 1)
        predicted_class_index = (prediction > 0.5).astype("int32")[0][0]

        return class_names[predicted_class_index]
    except Exception as e:
        return str(e)