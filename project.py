# Anthony Tran
# Date: 12/18/2024
# Assignment: Project

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing

# Data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,               # Normalize pixel values to [0, 1]
    validation_split=0.2,            # 20% of data for validation
    rotation_range=20,               # Randomly rotate images
    width_shift_range=0.1,           # Randomly shift images horizontally
    height_shift_range=0.1,          # Randomly shift images vertically
    horizontal_flip=True             # Randomly flip images horizontally
)

# Load training data
train_data = datagen.flow_from_directory(
    directory="data",
    target_size=(128, 128),          # Resize images to 128x128
    batch_size=32,                   # Batch size for training
    class_mode="categorical",        # Multi-class classification
    subset="training",               # Specify training subset
)

# Load validation data
val_data = datagen.flow_from_directory(
    directory="data",
    target_size=(128, 128),          # Resize images to 128x128
    batch_size=32,                   # Batch size for validation
    class_mode="categorical",        # Multi-class classification
    subset="validation",             # Specify validation subset
)

# Class labels mapping
print("Class indices:", train_data.class_indices)

# Step 2: Model Design
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Explicit input layer
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(8, activation="softmax"),
])

# Compile the model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Step 3: Train the Model
print("Training the model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # Adjust epochs based on performance
)

# Step 4: Evaluate the Model
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(val_data)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Step 5: Plot Training and Validation Accuracy/Loss
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.tight_layout()
plt.show()

# Step 6: Save the Model
model.save("natural_images_model.h5")
print("Model saved as 'natural_images_model.h5'.")
