import numpy as np
from torch.utils.data import DataLoader
from SpeakerDiarizationDNN import SpeakerDiarizationDNN
from Trainer import train_model
from DiarizationChunkDataset import DiarizationChunkDataset

audio_file = "010 - Die drei Fragezeichen und die fluesternde Mumie (A).mp3"
label_file = "010 - Die drei Fragezeichen und die fluesternde Mumie (A).txt"

class_names = "classes.txt"

dataset = DiarizationChunkDataset(
    audio_path=audio_file,
    label_path=label_file,
    class_path=class_names,
    frame_size=0.5,
    hop_size=0.25,
    sample_rate=16000
)

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
model = SpeakerDiarizationDNN(num_classes=dataset.get_num_classes())

train_model(model, train_loader, num_epochs=15, device='cpu')
