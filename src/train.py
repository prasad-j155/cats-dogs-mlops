import os
import tensorflow as tf
import json
import keras
from keras import layers, models, optimizers


import yaml

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)



# Define paths
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")

IMG_SIZE = (params["prepare"]["img_size"], params["prepare"]["img_size"])
BATCH_SIZE = params["train"]["batch_size"]
EPOCHS = params["train"]["epochs"]


"""
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5  # Keep it small for testing the pipeline
"""


def build_model():
    """Builds a simple Convolutional Neural Network."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary classification (Cat vs Dog)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    print("Loading datasets...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DIR, 'train'),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DIR, 'val'),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    print("Building model...")
    model = build_model()
    
    print("Starting training...")
    # Capture the training history
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
    
    # Extract the final epoch's validation metrics
    val_accuracy = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]

    # Save metrics to a JSON file
    metrics = {
        "accuracy": float(val_accuracy),
        "loss": float(val_loss)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")

    print(f"Saving model to {MODEL_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print("Training complete!")

if __name__ == "__main__":
    train()