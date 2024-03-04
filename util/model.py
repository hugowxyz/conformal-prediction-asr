import torch
import torch.nn as nn

import torchaudio
from torch.utils.data import Dataset
import numpy as np

# Define a simple CNN architecture
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(511616, 128)  # 246 is the output size after convolutions
        self.fc2 = nn.Linear(128, 4)  # 4 classes

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
# Define a dataset class for loading audio data
class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16_000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)(waveform)
        # Trim to 1 second
        waveform = waveform[:, :8000]
        # Pad if shorter than 1 second
        if waveform.size(1) < 8000:
            pad_amount = 8000 - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        # Convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, self.labels[idx]