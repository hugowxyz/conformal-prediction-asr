import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from model import AudioClassifier, AudioDataset

# Example usage
if __name__ == "__main__":
    root_path = "2023-12-04-HMI-dataset"
    folder_name = "agtbv"
    audio_paths = os.listdir(os.path.join(root_path, folder_name))
    audio_paths = [os.path.join(root_path, folder_name, file) for file in audio_paths]
    labels = []
    for file in audio_paths:
        filename = os.path.splitext(file)[0]
        if filename.endswith("cafe"):
            labels.append(1)
        elif filename.endswith("highway"):
            labels.append(2)
        elif filename.endswith("park"):
            labels.append(3)
        else:
            labels.append(0)
    
    # Create dataset
    dataset = AudioDataset(audio_paths, labels)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create model
    model = AudioClassifier()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(10):  # 5 epochs for demonstration
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    # Save the trained model
    torch.save(model.state_dict(), "audio_classifier_model.pth")
