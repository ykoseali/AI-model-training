# -*- coding: utf-8 -*-


import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

"""## Preparing The Dataset

### Check Dataset Integrity
"""

# Check dataset folder structure
def check_dataset_integrity():
    train_dir = "/content/dataset/train"
    validation_dir = "/content/dataset/validation"
    test_dir = "/content/dataset/test"

    for folder in [train_dir, validation_dir, test_dir]:
        if not os.path.exists(folder):
            print(f"Error: {folder} does not exist.")
            return

    # Check overlap
    for category in os.listdir(train_dir):
        train_files = set(os.listdir(os.path.join(train_dir, category)))
        validation_files = set(os.listdir(os.path.join(validation_dir, category)))
        test_files = set(os.listdir(os.path.join(test_dir, category)))

        train_val_overlap = train_files.intersection(validation_files)
        train_test_overlap = train_files.intersection(test_files)
        val_test_overlap = validation_files.intersection(test_files)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            print(f"Overlap detected in {category}:")
            print(f"Train-Validation Overlap: {train_val_overlap}")
            print(f"Train-Test Overlap: {train_test_overlap}")
            print(f"Validation-Test Overlap: {val_test_overlap}")
        else:
            print(f"No overlap found in {category}.")

check_dataset_integrity()

"""### Data Augmentation"""

datagen_train = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen_validation = ImageDataGenerator(rescale=1.0/255.0)
datagen_test = ImageDataGenerator(rescale=1.0/255.0)

# Load datasets
train_data = datagen_train.flow_from_directory(
    "/content/dataset/train",
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

validation_data = datagen_validation.flow_from_directory(
    "/content/dataset/validation",
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

test_data = datagen_test.flow_from_directory(
    "/content/dataset/test",
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

"""## Building The Model Using Transfer Learning"""

# Load MobileNetV2
base_model = MobileNetV2(input_shape=(100, 100, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Regularization to prevent overfitting
    Dense(train_data.num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

"""## Training The Model"""

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10,
    callbacks=[early_stopping]
)

"""## Visualizing Training and Validation Process"""

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

"""## Evaluating The Model"""

# Evaluate test accuracy
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Augmented test data
datagen_test_augmented = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=10,
    brightness_range=[0.9, 1.1],
    zoom_range=0.2
)

test_data_augmented = datagen_test_augmented.flow_from_directory(
    "/content/dataset/test",
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate on augmented test data
test_loss_aug, test_accuracy_aug = model.evaluate(test_data_augmented)
print(f"Augmented Test Accuracy: {test_accuracy_aug:.2f}")

# Predictions and true labels
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_data.class_indices.keys()))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Test Data)')
plt.show()

# Classification report
print("Classification Report (Test Data):")
print(classification_report(y_true, y_pred_classes, target_names=test_data.class_indices.keys()))

"""## Additional Testing of Model with New Images"""

from tensorflow.keras.preprocessing import image

# Load and preprocess a new image
img_path = "/content/images.jpeg"  # Replace with the path to a new image
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict the class
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
print(f"Predicted Class: {list(test_data.class_indices.keys())[predicted_class]}")