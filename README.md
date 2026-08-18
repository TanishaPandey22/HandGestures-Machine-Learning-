# ==========================================
# Hand Gesture Recognition using CNN
# Part 1 - Imports, Configuration,
# Dataset Loading & Preprocessing
# ==========================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "leapGestRecog"

IMAGE_SIZE = 64

BATCH_SIZE = 32

EPOCHS = 15

MODEL_PATH = "gesture_model.keras"

# ==========================================
# Gesture Labels
# ==========================================

gesture_names = {
    "01_palm": "Palm",
    "02_l": "L",
    "03_fist": "Fist",
    "04_fist_moved": "Fist Moved",
    "05_thumb": "Thumb",
    "06_index": "Index",
    "07_ok": "OK",
    "08_palm_moved": "Palm Moved",
    "09_c": "C",
    "10_down": "Down"
}

print("=" * 50)
print("Hand Gesture Recognition")
print("=" * 50)

# ==========================================
# Load Dataset
# ==========================================

images = []

labels = []

print("\nLoading Dataset...\n")

# Loop through every subject
for subject in sorted(os.listdir(DATASET_PATH)):

    subject_path = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_path):
        continue

    print(f"Processing Subject : {subject}")

    # Loop through gesture folders
    for gesture_folder in sorted(os.listdir(subject_path)):

        gesture_path = os.path.join(subject_path, gesture_folder)

        if not os.path.isdir(gesture_path):
            continue

        # Read all images
        for image_name in tqdm(os.listdir(gesture_path),
                               desc=gesture_folder):

            image_path = os.path.join(
                gesture_path,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # Resize image
            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Normalize pixels
            image = image / 255.0

            images.append(image)

            labels.append(gesture_folder)

print("\nDataset Loaded Successfully")

# ==========================================
# Convert to NumPy Arrays
# ==========================================

X = np.array(images, dtype="float32")

y = np.array(labels)

print("\nDataset Shape")

print("Images :", X.shape)

print("Labels :", y.shape)

# ==========================================
# Encode Labels
# ==========================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

num_classes = len(np.unique(y))

print("\nNumber of Classes :", num_classes)

# Convert to One-Hot Encoding

y = to_categorical(y, num_classes)

print("Encoded Labels Shape :", y.shape)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    shuffle=True,

    stratify=y

)

print("\nTraining Images :", X_train.shape)

print("Testing Images :", X_test.shape)

print("\nDataset Ready For Training")
print("=" * 50)
# ==========================================
# Part 2 - CNN Model, Training,
# Evaluation & Model Saving
# ==========================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==========================================
# Build CNN Model
# ==========================================

print("\nBuilding CNN Model...\n")

model = Sequential()

# -------- Block 1 --------
model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

# -------- Block 2 --------

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

# -------- Block 3 --------

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )
)

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

# -------- Dense Layers --------

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation="softmax"))

print(model.summary())

# ==========================================
# Compile Model
# ==========================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nModel Compiled Successfully!")

# ==========================================
# Callbacks
# ==========================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

# ==========================================
# Train Model
# ==========================================

print("\nTraining Started...\n")

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test, y_test),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[early_stop, checkpoint],

    verbose=1

)

print("\nTraining Completed Successfully!")

# ==========================================
# Evaluate Model
# ==========================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)

print(f"\nTest Loss     : {loss:.4f}")

print(f"Test Accuracy : {accuracy*100:.2f}%")

# ==========================================
# Predictions
# ==========================================

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

y_true = np.argmax(y_test, axis=1)

# ==========================================
# Accuracy
# ==========================================

acc = accuracy_score(y_true, y_pred)

print(f"\nAccuracy Score : {acc*100:.2f}%")

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(

    classification_report(

        y_true,

        y_pred,

        target_names=encoder.classes_

    )

)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(

    y_true,

    y_pred

)

print("\nConfusion Matrix\n")

print(cm)

# ==========================================
# Save Final Model
# ==========================================

model.save(MODEL_PATH)

print("\nModel Saved Successfully!")

print(f"Saved As : {MODEL_PATH}")

print("\nTraining Finished Successfully.")

# ==========================================
# Part 3 - Graphs & Confusion Matrix
# ==========================================

import seaborn as sns

# ==========================================
# Create Results Folder
# ==========================================

RESULTS_DIR = "results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# ==========================================
# Accuracy Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

accuracy_path = os.path.join(
    RESULTS_DIR,
    "accuracy.png"
)

plt.savefig(accuracy_path)

plt.show()

# ==========================================
# Loss Graph
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

loss_path = os.path.join(
    RESULTS_DIR,
    "loss.png"
)

plt.savefig(loss_path)

plt.show()

# ==========================================
# Confusion Matrix
# ==========================================

plt.figure(figsize=(10,8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=encoder.classes_,

    yticklabels=encoder.classes_

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

confusion_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(confusion_path)

plt.show()

print("\nGraphs Saved Successfully!")

print(f"Accuracy Graph       : {accuracy_path}")
print(f"Loss Graph           : {loss_path}")
print(f"Confusion Matrix     : {confusion_path}")

# ==========================================
# Final Message
# ==========================================

print("\n====================================")
print("Hand Gesture Recognition Completed")
print("====================================")

print(f"Final Test Accuracy : {accuracy*100:.2f}%")
print(f"Model Saved         : {MODEL_PATH}")
print(f"Results Folder      : {RESULTS_DIR}")