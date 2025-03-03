from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

def evaluate_model(model, test_loader):
    """
    Evaluates the model on the test dataset and prints accuracy and confusion matrix.
    """
    y_true, y_pred = [], []
    for audio_input, video_input, labels in test_loader:
        outputs = model(audio_input, video_input)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")