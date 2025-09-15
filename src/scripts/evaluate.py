import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def evaluate(model_path, validation_dir):
    # Load the model from the specified path
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Prepare ImageDataGenerator for validation data
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    # Load validation data from directory
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        shuffle=False
    )

    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")

    # Get predictions and true labels
    y_val = validation_generator.classes
    y_pred = model.predict(validation_generator, verbose=1)

    y_pred_class = (y_pred > 0.5).astype("int32")

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred_class))

    # ROC-AUC (Only applicable for binary classification)
    if validation_generator.num_classes == 2:
        roc_auc = roc_auc_score(y_val, y_pred)
        print(f"ROC-AUC Score: {roc_auc}")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_val, y_pred)
        pr_auc = auc(recall, precision)
        print(f"Precision-Recall AUC: {pr_auc}")