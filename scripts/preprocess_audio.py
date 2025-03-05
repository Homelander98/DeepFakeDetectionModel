import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from models.ensemble import EnsembleModel
from utils.data_loader import load_data
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
args = parser.parse_args()

# Initialize TensorBoard writer
writer = SummaryWriter("runs/experiment_name")

# Load data
train_loader, test_loader = load_data(batch_size=args.batch_size)

# Initialize model
model = EnsembleModel(audio_input_size=13, video_input_size=256, hidden_size=128, num_classes=2).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce learning rate every 5 epochs

# Mixed precision training
scaler = GradScaler()

# Save the best model
best_val_loss = float('inf')
best_model_path = "models/best_model.pth"
os.makedirs("models", exist_ok=True)

# Early stopping
early_stopping_patience = 3  # Stop if validation loss does not improve for 3 epochs
early_stopping_counter = 0

# Training loop
for epoch in range(args.epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

    for audio_input, video_input, labels in progress_bar:
        # Move data to device
        audio_input = audio_input.to(device)
        video_input = video_input.to(device)
        labels = labels.to(device)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(audio_input, video_input)
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update training loss
        train_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"Loss": loss.item()})

    # Log training loss
    writer.add_scalar("Training Loss", train_loss / len(train_loader), epoch)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for audio_input, video_input, labels in test_loader:
            # Move data to device
            audio_input = audio_input.to(device)
            video_input = video_input.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(audio_input, video_input)
            loss = criterion(outputs, labels)

            # Update validation loss
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    # Log validation metrics
    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predicted, average='binary')
    recall = recall_score(all_labels, all_predicted, average='binary')
    f1 = f1_score(all_labels, all_predicted, average='binary')
    writer.add_scalar("Validation Loss", val_loss, epoch)
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)
    writer.add_scalar("Precision", precision, epoch)
    writer.add_scalar("Recall", recall, epoch)
    writer.add_scalar("F1-Score", f1, epoch)

    print(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {train_loss / len(train_loader):.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

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