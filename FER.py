from google.colab import drive
drive.mount('/content/drive')
# Import necessary libraries
import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Dataset path
datasetPath = "/content/drive/MyDrive/facial-expression-dataset/images/new_validation"

# Image dimensions and lists to store images and labels
imageSize = (48, 48)
X_train = []
y_train = []

# Define emotion mapping based on folder names in the new dataset
emotion_label_mapping = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

# Read the dataset, resize images, and convert to grayscale
for emotion_folder in os.listdir(datasetPath + "/train"):
    emotion_path = os.path.join(datasetPath + "/train", emotion_folder)
    if emotion_folder not in emotion_label_mapping:
        print(f"Skipping folder '{emotion_folder}' - Not in emotion mapping.")
        continue

    if os.path.isdir(emotion_path):
        for image_file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, image_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, imageSize)
                    X_train.append(img_resized)
                    y_train.append(emotion_label_mapping[emotion_folder])
                else:
                    print(f"Error loading image: {img_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(emotion_label_mapping))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Reshape X_train to include the channel dimension
X_train = X_train.reshape(X_train.shape[0], imageSize[0], imageSize[1], 1)

# Reshape X_val to include the channel dimension as well
X_val = X_val.reshape(X_val.shape[0], imageSize[0], imageSize[1], 1)

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Build a deeper CNN model with Batch Normalization
model = Sequential([
    layers.InputLayer(input_shape=(imageSize[0], imageSize[1], 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(emotion_label_mapping), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Train the model with data augmentation
history = model.fit(data_gen.flow(X_train, y_train, batch_size=64),
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, lr_reduction])

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('/content/drive/MyDrive/facial_expression_model.keras')

print("Model has been trained and saved successfully.")

from sklearn.metrics import confusion_matrix

# Assuming you have y_pred (predicted labels) and y_true (true labels)
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)


cm = confusion_matrix(y_true_classes, y_pred_classes)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Optionally, you can visualize the confusion matrix using seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()