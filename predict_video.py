import numpy as np
import cv2
import json
from collections import deque
from tensorflow.keras.models import load_model

# Load model and classes
model = load_model("models/cnn_action.h5")

with open("models/class_indices.json") as f:
    class_indices = json.load(f)

# IMPORTANT: keep the exact order used by Keras
classes = list(class_indices.keys())

# Video
cap = cv2.VideoCapture("test4.mp4")

K = 32
prob_queue = deque(maxlen=K)
frame_count = 0
skip_frames = 5   # process 1 frame every 5 to go faster

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames for speed
    if frame_count % skip_frames != 0:
        continue

    # Preprocessing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    probs = model.predict(img, verbose=0)[0]
    prob_queue.append(probs)

    # Use smoothing only when queue is full
    if len(prob_queue) == K:
        avg_probs = np.mean(prob_queue, axis=0)
        pred_class = classes[np.argmax(avg_probs)]
    else:
        pred_class = "..."

    # Draw label on frame
    cv2.putText(frame, f"Action: {pred_class}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA)

    cv2.imshow("Video Action Recognition", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
