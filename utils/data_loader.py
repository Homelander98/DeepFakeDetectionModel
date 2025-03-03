import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys

def load_data():
    # Load preprocessed video and audio features
    try:
        video_features = np.load("data/processed_frames/video_features.npy")
        audio_features = np.load("data/audio_features/audio_features.npy")
        labels = np.load("data/labels.npy")  # Ensure labels are saved during preprocessing
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the preprocessing scripts have been run.")
        sys.exit(1)

    # Convert to PyTorch tensors
    video_tensor = torch.tensor(video_features, dtype=torch.float32)
    audio_tensor = torch.tensor(audio_features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(audio_tensor, video_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader