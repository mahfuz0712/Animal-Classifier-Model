import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import datetime

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

def train(train_dir, validation_dir):
    """
    Trains a convolutional neural network (CNN) model for image classification.
    
    Args:
        train_dir (str): Path to the training data directory.
        validation_dir (str): Path to the validation data directory.
    """
    # Data generators with data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, 
        rotation_range=40, 
        width_shift_range=0.2,
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True
    )
    # Rescaling generator for the validation set
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    # Use flow_from_directory to create a training data generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    # Use flow_from_directory to create a validation data generator
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    # Define the model architecture using the Sequential API
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model with an optimizer, loss function, and metrics
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    # Train the model using the generators
    print("Starting model training...")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator
    )

    # Save the trained model with a unique timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join('models', f'acm_{timestamp}.keras')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")