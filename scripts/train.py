import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.ensemble import EnsembleModel
from utils.data_loader import load_data
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_loader, test_loader = load_data()

# Initialize model
model = EnsembleModel(audio_input_size=13, video_input_size=256, hidden_size=128, num_classes=2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce learning rate every 5 epochs

# Save the best model
best_val_loss = float('inf')
best_model_path = "models/best_model.pth"
os.makedirs("models", exist_ok=True)

# Early stopping
early_stopping_patience = 3  # Stop if validation loss does not improve for 3 epochs
early_stopping_counter = 0

# Training loop
for epoch in range(10):
    model.train()  # Set model to training mode
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10", leave=False)

    for audio_input, video_input, labels in progress_bar:
        # Move data to device
        audio_input = audio_input.to(device)
        video_input = video_input.to(device)
        labels = labels.to(device)

        # Ensure audio_input has 3 dimensions: (batch_size, sequence_length=1, input_size)
        if audio_input.dim() == 2:
            audio_input = audio_input.unsqueeze(1)

        # Print input shapes for debugging
        print("Audio input shape:", audio_input.shape)
        print("Video input shape:", video_input.shape)

        # Forward pass
        outputs = model(audio_input, video_input)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"Loss": loss.item()})

    # Average training loss for the epoch
    train_loss /= len(train_loader)
    print(f"Epoch [{epoch+1}/10], Training Loss: {train_loss:.4f}")

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for audio_input, video_input, labels in test_loader:
            # Move data to device
            audio_input = audio_input.to(device)
            video_input = video_input.to(device)
            labels = labels.to(device)

            # Ensure audio_input has 3 dimensions: (batch_size, sequence_length=1, input_size)
            if audio_input.dim() == 2:
                audio_input = audio_input.unsqueeze(1)

            # Ensure video_input has 3 dimensions: (batch_size, sequence_length=1, input_size)
            if video_input.dim() == 2:
                video_input = video_input.unsqueeze(1)

            # Forward pass
            outputs = model(audio_input, video_input)
            loss = criterion(outputs, labels)

            # Update validation loss
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Average validation loss and accuracy
    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/10], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with validation loss: {val_loss:.4f}")
        early_stopping_counter = 0  # Reset early stopping counter
    else:
        early_stopping_counter += 1

    # Early stopping
    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}.")
        break

    # Update learning rate
    scheduler.step()

print("Training complete. Best model saved to:", best_model_path)