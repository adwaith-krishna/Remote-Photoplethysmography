import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time

# Constants
FACE_DETECTION_MODEL = 'haarcascade_frontalface_default.xml'
MODEL_PATH = 'modele500.h5'
SEQUENCE_LENGTH = 30
FRAME_RATE = 30
WINDOW_SIZE = 1.0
IMAGE_SIZE = (64, 64)
MIN_BPM = 40  # Physiologically reasonable minimum
MAX_BPM = 180  # Physiologically reasonable maximum

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_DETECTION_MODEL)
model = load_model(MODEL_PATH)

# Initialize buffers
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
bpm_buffer = deque(maxlen=int(FRAME_RATE * WINDOW_SIZE))
last_time = time.time()
last_valid_bpm = 72  # Default reasonable value


def extract_and_preprocess_roi(frame):
    """Extract face ROI with validation checks"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]

    # Validate face detection parameters
    if w < 100 or h < 100:  # Minimum face size
        return None

    roi = frame[y:y + h, x:x + w]

    try:
        resized = cv2.resize(roi, IMAGE_SIZE)
        normalized = resized.astype('float32') / 255.0

        # Validate pixel values
        if np.isnan(normalized).any() or np.isinf(normalized).any():
            return None

        return normalized, (x, y, w, h)
    except:
        return None


def process_frames(frames):
    """Process frame sequence with robust prediction"""
    try:
        frames_array = np.array(frames)

        # Validate the frame sequence
        if (np.isnan(frames_array).any() or
                np.isinf(frames_array).any() or
                frames_array.shape != (SEQUENCE_LENGTH, *IMAGE_SIZE, 3)):
            return None

        frames_array = np.expand_dims(frames_array, axis=0)
        prediction = model.predict(frames_array, verbose=0)

        # Clip prediction to physiologically reasonable range
        bpm = np.clip(prediction[0][0], MIN_BPM, MAX_BPM)

        # Additional validation
        if np.isnan(bpm) or np.isinf(bpm):
            return None

        return float(bpm)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None


# Main processing loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time()

    # Extract and preprocess ROI
    result = extract_and_preprocess_roi(frame)

    if result is not None:
        processed_frame, (x, y, w, h) = result
        frame_buffer.append(processed_frame)

        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Predict when buffer is full
        if len(frame_buffer) == SEQUENCE_LENGTH:
            current_bpm = process_frames(frame_buffer)

            if current_bpm is not None:
                last_valid_bpm = current_bpm
                bpm_buffer.append(current_bpm)
                smoothed_bpm = np.mean(bpm_buffer)

                # Display with confidence indicator
                confidence = min(len(bpm_buffer) / bpm_buffer.maxlen, 1.0)
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.putText(frame, f"HR: {smoothed_bpm:.1f} BPM", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                # Display last valid with warning
                cv2.putText(frame, f"HR: {last_valid_bpm:.1f} BPM (estimating)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    # Display FPS and buffer status
    fps = 1 / (current_time - last_time)
    last_time = current_time
    buffer_status = len(frame_buffer) / SEQUENCE_LENGTH

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Buffer: {buffer_status:.0%}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, int(255 * buffer_status), int(255 * (1 - buffer_status))), 2)

    cv2.imshow('Heart Rate Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()