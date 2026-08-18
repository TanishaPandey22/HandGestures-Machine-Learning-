import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "gesture_model.keras"

IMAGE_PATH = "test_image.jpg"     # Change this to your image

IMAGE_SIZE = 64

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
# Load Model
# ==========================================

print("Loading Model...")

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!\n")

# ==========================================
# Read Image
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:

    print("Error: Image not found!")

    exit()

# Keep original image

original = image.copy()

# ==========================================
# Preprocess Image
# ==========================================

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

image = image.astype("float32") / 255.0

image = np.expand_dims(image, axis=0)

# ==========================================
# Prediction
# ==========================================

prediction = model.predict(image)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction)

gesture = gesture_labels[predicted_class]

# ==========================================
# Output
# ==========================================

print("=" * 40)

print("Predicted Gesture :", gesture)

print(f"Confidence : {confidence*100:.2f}%")

print("=" * 40)

# ==========================================
# Show Image
# ==========================================

text = f"{gesture} ({confidence*100:.1f}%)"

cv2.putText(

    original,

    text,

    (20, 40),

    cv2.FONT_HERSHEY_SIMPLEX,

    1,

    (0, 255, 0),

    2

)

cv2.imshow("Prediction", original)

cv2.waitKey(0)

cv2.destroyAllWindows()