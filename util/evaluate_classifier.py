import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from model import AudioClassifier, AudioDataset
from load_dataset import load_dataset

model = AudioClassifier()
model.load_state_dict(torch.load("audio_classifier_model.pth"))
model.eval()  # Set the model to evaluation mode

audio_paths, labels = load_dataset("../2023-12-04-HMI-dataset")

 # Create dataset
dataset = AudioDataset(audio_paths[:500], labels)

# Create data loader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

correct = 0
total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Append true and predicted labels for F1 score calculation
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())


accuracy = correct / total
print("Accuracy:", accuracy)

# Calculate F1 score
f1 = f1_score(y_true, y_pred, average='weighted')
print("F1 Score:", f1)

