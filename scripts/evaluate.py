import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import os

# Disable oneDNN optimizations if necessary
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def evaluate_model():
    validation_dir = "../data/validation"
    # Load the model
    model = tf.keras.models.load_model('../models/animal_model.keras')
    print("Model loaded successfully.")


    # Prepare ImageDataGenerator for validation data
    validation_datagen = ImageDataGenerator(rescale=1.0/255) # Normalize pixel values to [0, 1]

    # Load validation data from directory
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),  # Resize images to the model's input size
        batch_size=20,
        class_mode='binary',  # Use 'binary' for binary classification or 'categorical' for multi-class
        shuffle=False  # Don't shuffle the validation data to keep predictions aligned with true labels
    )

    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")

    # Get predictions for classification report and other metrics
    y_val = validation_generator.classes  # True labels
    y_pred = model.predict(validation_generator, verbose=1)  # Predictions from the model

    # If output is a single probability (sigmoid output), classify as 0 or 1
    y_pred_class = (y_pred > 0.5).astype("int32")  # Convert probabilities to class labels (0 or 1)

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred_class))

    # ROC-AUC (Only applicable for binary classification)
    if validation_generator.num_classes == 2:  # Binary classification
        # For binary output, the model predicts a single probability value
        roc_auc = roc_auc_score(y_val, y_pred)  # Use the probabilities for the positive class (single column)
        print(f"ROC-AUC Score: {roc_auc}")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_val, y_pred)  # Use the single probability values
        pr_auc = auc(recall, precision)
        print(f"Precision-Recall AUC: {pr_auc}")

# Call the function to evaluate the model
evaluate_model()