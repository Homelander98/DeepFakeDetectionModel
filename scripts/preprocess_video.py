import sys
import os
import cv2
from mtcnn import MTCNN
import glob
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def split_video_to_frames(video_path, output_folder):
    """Split a video into frames and save them to the output folder."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video name without extension
    frame_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Save frames for feature extraction
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}.")
    return frames

def detect_and_crop_faces(frame, detector):
    """Detect and crop faces in a frame using MTCNN."""
    faces = detector.detect_faces(frame)
    cropped_faces = []
    for face in faces:
        x, y, width, height = face['box']
        cropped_face = frame[y:y+height, x:x+width]
        cropped_faces.append(cropped_face)
    return cropped_faces

def resize_faces(faces, target_size=(128, 128)):
    """Resize all cropped faces to a consistent size."""
    resized_faces = []
    for face in faces:
        resized_face = cv2.resize(face, target_size)  # Resize face to target size
        resized_faces.append(resized_face)
    return resized_faces

def compute_optical_flow_gpu(prev_frame, next_frame):
    """Compute optical flow between two frames using GPU."""
    # Convert frames to PyTorch tensors and move to GPU
    prev_tensor = to_tensor(prev_frame).unsqueeze(0).to(device)
    next_tensor = to_tensor(next_frame).unsqueeze(0).to(device)

    # Compute optical flow using PyTorch (example using a simple difference)
    flow = next_tensor - prev_tensor
    return flow.cpu().squeeze(0)  # Move back to CPU for saving/display

def save_video_features(video_features, output_path):
    """
    Save video features, maintaining video-level separation.
    
    Args:
        video_features (list): A list of lists, where each inner list contains face arrays for a video
        output_path (str): Path to save the features
    """
    # Use object dtype to handle different-sized arrays
    features_array = np.array(video_features, dtype=object)
    
    # Save the features
    np.save(output_path, features_array)
    print(f"Saved features for {len(features_array)} videos to {output_path}.")

def process_videos(video_folder, output_folder, detector, max_videos=5):
    """Process videos in a folder and return video features."""
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))  # Adjust the extension if needed
    video_features = []  # List to store features for all videos
    processed_count = 0

    for video_path in video_files:
        if processed_count >= max_videos:
            break  # Stop after processing max_videos

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_folder = os.path.join(output_folder, video_name)

        # Skip if video has already been processed
        if os.path.exists(video_output_folder):
            print(f"Skipping {video_name}: Already processed.")
            continue

        print(f"Processing video: {video_name}")

        # Step 1: Split video into frames
        frames = split_video_to_frames(video_path, video_output_folder)
        if not frames:
            print(f"Warning: No frames extracted from {video_name}. Skipping.")
            continue

        # Step 2: Detect and crop faces in each frame
        cropped_faces = []
        for frame in frames:
            faces = detect_and_crop_faces(frame, detector)
            cropped_faces.extend(faces)  # Save all cropped faces

        # Step 3: Resize cropped faces to a consistent size
        resized_faces = resize_faces(cropped_faces, target_size=(128, 128))  # Resize faces to 128x128

        # Step 4: Compute optical flow for all consecutive frames (GPU-accelerated)
        if len(frames) > 1:
            for i in range(len(frames) - 1):
                flow = compute_optical_flow_gpu(frames[i], frames[i + 1])
                print(f"Optical flow computed for frames {i} and {i + 1}.")

        # Save video features (e.g., resized cropped faces)
        video_features.append(resized_faces)  # Example: Save resized faces as features
        processed_count += 1

    return video_features

if __name__ == "__main__":
    # Paths
    fake_video_folder = r"C:\Users\SIAM\Desktop\deepfake_detection\data\Celeb-synthesis"  # Fake videos
    real_video_folder = r"C:\Users\SIAM\Desktop\deepfake_detection\data\Celeb-real"  # Real videos
    output_folder = "data/processed_frames"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize face detector
    detector = MTCNN()

    # Process fake videos first
    print("Processing fake videos...")
    fake_video_features = process_videos(fake_video_folder, os.path.join(output_folder, "fake"), detector, max_videos=5)

    # Process real videos next
    print("Processing real videos...")
    real_video_features = process_videos(real_video_folder, os.path.join(output_folder, "real"), detector, max_videos=5)

    # Combine features from fake and real videos
    video_features = fake_video_features + real_video_features

    # Save all video features to a single file
    save_video_features(video_features, os.path.join(output_folder, "video_features.npy"))