# import sys
# import os
# import cv2
# from mtcnn import MTCNN
# import glob
# import torch
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_tensor, to_pil_image

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# def split_video_to_frames(video_path, output_folder):
#     """Split a video into frames and save them to the output folder."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}.")
#         return 0

#     video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video name without extension
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imwrite(f"{output_folder}/{video_name}_frame_{frame_count:04d}.jpg", frame)  # Include video name
#         frame_count += 1

#     cap.release()
#     print(f"Saved {frame_count} frames to {output_folder}.")
#     return frame_count

# def detect_and_crop_faces(frame, detector):
#     """Detect and crop faces in a frame using MTCNN."""
#     faces = detector.detect_faces(frame)
#     cropped_faces = []
#     for face in faces:
#         x, y, width, height = face['box']
#         cropped_face = frame[y:y+height, x:x+width]
#         cropped_faces.append(cropped_face)
#     return cropped_faces

# def compute_optical_flow_gpu(prev_frame, next_frame):
#     """Compute optical flow between two frames using GPU."""
#     # Convert frames to PyTorch tensors and move to GPU
#     prev_tensor = to_tensor(prev_frame).unsqueeze(0).to(device)
#     next_tensor = to_tensor(next_frame).unsqueeze(0).to(device)

#     # Compute optical flow using PyTorch (example using a simple difference)
#     flow = next_tensor - prev_tensor
#     return flow.cpu().squeeze(0)  # Move back to CPU for saving/display

# def save_processed_video(frames, output_path, fps=30):
#     """Save a list of frames as a video."""
#     if not frames:
#         print("Error: No frames to save.")
#         return

#     height, width, _ = frames[0].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     for frame in frames:
#         out.write(frame)
#     out.release()
#     print(f"Saved processed video to {output_path}.")

# if __name__ == "__main__":
#     # Paths
#     video_folder = r"C:\Users\SIAM\Desktop\deepfake_detection\data\Celeb-synthesis"  # Use raw string for Windows paths
#     output_folder = "data/processed_frames/fake"
#     os.makedirs(output_folder, exist_ok=True)

#     # Get all video files in the folder
#     video_files = glob.glob(os.path.join(video_folder, "*.mp4"))  # Adjust the extension if needed

#     if not video_files:
#         sys.exit("Error: No video files found in the folder. Exiting.")

#     # Initialize face detector
#     detector = MTCNN()

#     # Process each video
#     for video_path in video_files:
#         print(f"Processing video: {video_path}")

#         # Step 1: Split video into frames
#         frame_count = split_video_to_frames(video_path, output_folder)
#         if frame_count == 0:
#             print(f"Warning: No frames extracted from {video_path}. Skipping.")
#             continue

#         # Step 2: Detect and crop faces in each frame
#         frames = []
#         cropped_faces = []

#         for i in range(frame_count):
#             frame_path = f"{output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}_frame_{i:04d}.jpg"
#             if not os.path.exists(frame_path):
#                 print(f"Warning: Frame {frame_path} not found. Skipping.")
#                 continue

#             frame = cv2.imread(frame_path)
#             if frame is None:
#                 print(f"Warning: Could not read frame {frame_path}. Skipping.")
#                 continue

#             frames.append(frame)
#             faces = detect_and_crop_faces(frame, detector)
#             cropped_faces.extend(faces)  # Save all cropped faces

#         if not frames:
#             print(f"Warning: No valid frames found for {video_path}. Skipping.")
#             continue

#         # Step 3: Compute optical flow for all consecutive frames (GPU-accelerated)
#         if len(frames) > 1:
#             for i in range(len(frames) - 1):
#                 flow = compute_optical_flow_gpu(frames[i], frames[i + 1])
#                 print(f"Optical flow computed for frames {i} and {i + 1}.")

#         # Step 4: Save processed video (example with cropped faces)
#         if cropped_faces:
#             save_processed_video(cropped_faces, f"{output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}_processed.mp4")
#         else:
#             print("Warning: No cropped faces to save.")

#         # Optional: Save cropped faces
#         if cropped_faces:
#             cropped_faces_folder = f"{output_folder}/cropped_faces"
#             os.makedirs(cropped_faces_folder, exist_ok=True)
#             for i, face in enumerate(cropped_faces):
#                 cv2.imwrite(f"{cropped_faces_folder}/{os.path.splitext(os.path.basename(video_path))[0]}_face_{i:04d}.jpg", face)
#             print(f"Saved {len(cropped_faces)} cropped faces.")

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

def compute_optical_flow_gpu(prev_frame, next_frame):
    """Compute optical flow between two frames using GPU."""
    # Convert frames to PyTorch tensors and move to GPU
    prev_tensor = to_tensor(prev_frame).unsqueeze(0).to(device)
    next_tensor = to_tensor(next_frame).unsqueeze(0).to(device)

    # Compute optical flow using PyTorch (example using a simple difference)
    flow = next_tensor - prev_tensor
    return flow.cpu().squeeze(0)  # Move back to CPU for saving/display

def save_video_features(video_features, output_path):
    """Save video features to a file."""
    np.save(output_path, video_features)
    print(f"Saved video features to {output_path}.")

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

        # Step 3: Compute optical flow for all consecutive frames (GPU-accelerated)
        if len(frames) > 1:
            for i in range(len(frames) - 1):
                flow = compute_optical_flow_gpu(frames[i], frames[i + 1])
                print(f"Optical flow computed for frames {i} and {i + 1}.")

        # Save video features (e.g., cropped faces or optical flow)
        video_features.append(cropped_faces)  # Example: Save cropped faces as features
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