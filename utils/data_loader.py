import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2

class VideoFaceDataset(Dataset):
    """
    Custom Dataset for handling video face features and audio features
    """
    def __init__(self, video_features, audio_features, labels):
        """
        Args:
            video_features (np.ndarray): Array of video feature arrays
            audio_features (np.ndarray): Array of audio feature arrays
            labels (np.ndarray): Corresponding labels for the features
        """
        # Preprocess video features into a consistent tensor format
        self.video_features = torch.tensor(self.preprocess_features(video_features), dtype=torch.float32)
        
        # Preprocess audio features into a consistent tensor format
        self.audio_features = torch.tensor(audio_features, dtype=torch.float32)
        
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def preprocess_features(self, features):
        """
        Preprocess face feature arrays into a consistent format
        
        Args:
            features (np.ndarray): Array of face feature arrays
        
        Returns:
            np.ndarray: Processed features with consistent shape
        """
        # List to store processed features
        processed_features = []
        
        # Iterate through each video's face features
        for video_faces in features:
            for face in video_faces:
                # Ensure face is a NumPy array
                face_array = np.array(face)
                
                # Convert to grayscale if it's a color image
                if len(face_array.shape) == 3:
                    face_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
                
                # Ensure consistent size (128x128)
                face_array = cv2.resize(face_array, (128, 128))
                
                # Normalize pixel values
                face_array = face_array / 255.0
                
                processed_features.append(face_array)
        
        return np.array(processed_features)
    
    def __len__(self):
        return len(self.video_features)
    
    def __getitem__(self, idx):
        return self.audio_features[idx], self.video_features[idx], self.labels[idx]

def load_data():
    """
    Load preprocessed video and audio features and create data loaders
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Paths for features and labels
    video_features_path = "data/processed_frames/video_features.npy"
    audio_features_path = "data/audio_features.npy"  # Add path to audio features
    labels_path = "data/labels.npy"
    
    try:
        # Load video features, audio features, and labels
        video_features = np.load(video_features_path, allow_pickle=True)
        audio_features = np.load(audio_features_path, allow_pickle=True)  # Load audio features
        labels = np.load(labels_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the preprocessing scripts have been run.")
        sys.exit(1)
    
    # Create dataset
    dataset = VideoFaceDataset(video_features, audio_features, labels)
    
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# Optional: Function to generate labels if not already created
def generate_labels(video_features):
    """
    Generate labels based on the source of video features
    
    Args:
        video_features (np.ndarray): Array of video features
    
    Returns:
        np.ndarray: Labels (0 for real, 1 for fake)
    """
    labels = []
    for i, video_group in enumerate(video_features):
        # Assuming first half are real videos, second half are fake
        # Adjust this logic based on your actual data structure
        group_label = 1 if i >= len(video_features) // 2 else 0
        labels.extend([group_label] * len(video_group))
    
    return np.array(labels)

# Optional: Function to generate dummy audio features
def generate_dummy_audio_features(video_features):
    """
    Generate dummy audio features (e.g., zeros or random values)
    
    Args:
        video_features (np.ndarray): Array of video features
    
    Returns:
        np.ndarray: Dummy audio features
    """
    # Generate dummy audio features (e.g., 13 MFCC coefficients)
    num_samples = sum(len(video_group) for video_group in video_features)
    return np.random.rand(num_samples, 13)  # Example: Random audio features

# If labels or audio features haven't been saved, generate them
if __name__ == "__main__":
    try:
        labels = np.load("data/labels.npy")
    except FileNotFoundError:
        video_features = np.load("data/processed_frames/video_features.npy", allow_pickle=True)
        labels = generate_labels(video_features)
        np.save("data/labels.npy", labels)
        print("Generated and saved labels.")
    
    try:
        audio_features = np.load("data/audio_features.npy")
    except FileNotFoundError:
        video_features = np.load("data/processed_frames/video_features.npy", allow_pickle=True)
        audio_features = generate_dummy_audio_features(video_features)
        np.save("data/audio_features.npy", audio_features)
        print("Generated and saved dummy audio features.")