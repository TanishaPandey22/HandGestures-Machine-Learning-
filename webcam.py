import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "gesture_model.keras"
IMAGE_SIZE = 64

# ==========================================
# Load Model
# ==========================================

print("Loading Model...")

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ==========================================
# Gesture Labels
# ==========================================

gesture_labels = {
    0: "Palm",
    1: "L",
    2: "Fist",
    3: "Fist Moved",
    4: "Thumb",
    5: "Index",
    6: "OK",
    7: "Palm Moved",
    8: "C",
    9: "Down"
}

# ==========================================
# Open Webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("\nPress 'q' to quit.\n")

# ==========================================
# Webcam Loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Copy frame
    display = frame.copy()

    # --------------------------------------
    # Define ROI (Region of Interest)
    # --------------------------------------

    x1, y1 = 150, 80
    x2, y2 = 450, 380

    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]

    # --------------------------------------
    # Preprocess ROI
    # --------------------------------------

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    roi_rgb = cv2.resize(roi_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    roi_rgb = roi_rgb.astype("float32") / 255.0

    roi_rgb = np.expand_dims(roi_rgb, axis=0)

    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(roi_rgb, verbose=0)

    class_index = np.argmax(prediction)

    confidence = np.max(prediction)

    gesture = gesture_labels[class_index]

    # --------------------------------------
    # Display Prediction
    # --------------------------------------

    cv2.putText(
        display,
        f"Gesture : {gesture}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Confidence : {confidence*100:.2f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        display,
        "Place your hand inside the green box",
        (20, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # --------------------------------------
    # Show Webcam
    # --------------------------------------

    cv2.imshow("Hand Gesture Recognition", display)

    # Press Q to Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()