import os
import cv2
import numpy as np
from scipy.interpolate import interp1d

# Step 1: Extract frames from video
def extract_frames(video_path, output_folder):
    """
    Extracts frames from a video and saves them as images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")


# Step 2: Load ground truth heart rate data
def load_ground_truth(ground_truth_path):
    """
    Loads ground truth data from the text file and extracts heart rate.
    Assumes the first column is the PPG signal.
    """
    heart_rates = []
    with open(ground_truth_path, 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()
            if len(columns) > 0:
                # Use the first column as the PPG signal (for example)
                ppg_value = float(columns[0])
                heart_rates.append(ppg_value)
    return heart_rates


# Step 3: Resample heart rate data to match the number of frames
def resample_heart_rates(heart_rates, target_length):
    """
    Resamples heart rate data to match the target length using interpolation.
    """
    original_indices = np.arange(len(heart_rates))
    target_indices = np.linspace(0, len(heart_rates) - 1, target_length)
    interpolator = interp1d(original_indices, heart_rates, kind='linear')
    resampled_heart_rates = interpolator(target_indices)
    return resampled_heart_rates


# Step 4: Detect and crop face from a frame
def detect_and_crop_face(frame):
    """
    Detects and crops the face region from a frame using Haar Cascade.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return frame[y:y+h, x:x+w]  # Return the face region
    return None


# Step 5: Preprocess a frame (resize and normalize)
def preprocess_frame(frame, size=(64, 64)):  # Reduced resolution to 64x64
    """
    Resizes and normalizes a frame.
    """
    frame = cv2.resize(frame, size)  # Resize
    frame = frame / 255.0  # Normalize pixel values to [0, 1]
    return frame.astype(np.float32)  # Use float32 to save memory


# Step 6: Create sequences of frames and corresponding heart rates
def create_sequences(frames, heart_rates, sequence_length=30):
    """
    Organizes frames into sequences and pairs them with heart rates.
    """
    X, y = [], []

    # Ensure there are enough frames and heart rates for at least one sequence
    if len(frames) < sequence_length or len(heart_rates) < sequence_length:
        print(f"Skipping subject: Not enough frames or heart rates for sequence length {sequence_length}.")
        return np.array([]), np.array([])

    for i in range(len(frames) - sequence_length):
        X.append(frames[i:i+sequence_length])  # Sequence of frames
        y.append(heart_rates[i+sequence_length])  # Corresponding heart rate

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  # Use float32


# Step 7: Preprocess the entire dataset
d


# Main function to preprocess the dataset
if __name__ == "__main__":
    # Paths
    dataset_path = "datasets/UBFC2"  # Path to the UBFC dataset
    output_folder = "preprocessed_data"  # Folder to save preprocessed data

    # Preprocess the dataset
    preprocess_ubfc_dataset(dataset_path, output_folder, sequence_length=30, batch_size=20)