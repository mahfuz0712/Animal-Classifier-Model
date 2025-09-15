# main.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QMessageBox

# Import your scripts
from scripts.train import train
from scripts.evaluate import evaluate
from scripts.predict import predict_image

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Actions')
        self.setGeometry(100, 100, 300, 200)

        self.layout = QVBoxLayout()

        self.train_btn = QPushButton('Train')
        self.evaluate_btn = QPushButton('Evaluate')
        self.predict_btn = QPushButton('Predict')

        self.train_btn.clicked.connect(self.train_model)
        self.evaluate_btn.clicked.connect(self.evaluate_model)
        self.predict_btn.clicked.connect(self.predict_data)

        self.layout.addWidget(self.train_btn)
        self.layout.addWidget(self.evaluate_btn)
        self.layout.addWidget(self.predict_btn)

        self.setLayout(self.layout)

    def train_model(self):
        print("Opening file dialog for training data...")
        # Get training data directory
        train_dir = QFileDialog.getExistingDirectory(self, "Select Training Data Directory")
        if not train_dir:
            return  # User canceled

        print("Opening file dialog for validation data...")
        # Get validation data directory
        validation_dir = QFileDialog.getExistingDirectory(self, "Select Validation Data Directory")
        if not validation_dir:
            return  # User canceled

        try:
            print(f"Training model with data from: {train_dir} and {validation_dir}")
            train(train_dir, validation_dir)  # Pass paths to the train function
            QMessageBox.information(self, "Success", "Model training complete!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during training: {e}")

    def evaluate_model(self):
        print("Opening file dialog for model file and validation data...")
        # Get model file path
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "", "Keras Files (*.keras *.h5)")
        if not model_path:
            return # User canceled
        
        # Get validation data directory
        validation_dir = QFileDialog.getExistingDirectory(self, "Select Validation Data Directory")
        if not validation_dir:
            return # User canceled

        try:
            print(f"Evaluating model from: {model_path} with data from: {validation_dir}")
            evaluate(model_path, validation_dir) # Pass paths to the evaluate function
            QMessageBox.information(self, "Success", "Model evaluation complete!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during evaluation: {e}")

    def predict_data(self):
        print("Opening file dialog for model file and image...")
        # Get model file path
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "", "Keras Files (*.keras *.h5)")
        if not model_path:
            return # User canceled
        
        # Get image file path
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image to Predict", "", "Image Files (*.png *.jpg *.jpeg)")
        if not image_path:
            return # User canceled

        try:
            print(f"Predicting with model: {model_path} on image: {image_path}")
            # The predict function needs to be updated to accept the model path
            result = predict_image(model_path, image_path)
            QMessageBox.information(self, "Prediction Result", f"The image is classified as: {result}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())