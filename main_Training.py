import os.path
import torch
from torch.utils.data import DataLoader
from SpeakerDiarizationDNN import SpeakerDiarizationDNN
from Trainer import train_model
from DiarizationChunkDataset import DiarizationChunkDataset


def optimizer_to(optim, device):
    # suggested by user aaniin on the pytorch forum
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


audio_file = "010 - Die drei Fragezeichen und die fluesternde Mumie (A).mp3"
label_file = "010 - Die drei Fragezeichen und die fluesternde Mumie (A).txt"

class_names = "classes.txt"

filename_model = "ddf_diarizer"

device = "cuda"
learning_rate = 0.001
num_epochs = 10

dataset = DiarizationChunkDataset(
    audio_path=audio_file,
    label_path=label_file,
    class_path=class_names,
    frame_size=0.5,
    hop_size=0.25,
    sample_rate=16000
)

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

if not os.path.exists('Trained'):
    os.mkdir('Trained')

model = SpeakerDiarizationDNN(num_classes=dataset.get_num_classes())
if os.path.isfile(f'Trained/{filename_model}.pth'):
    sd = torch.load(f'Trained/{filename_model}.pth', weights_only=False)
    model.load_state_dict(sd)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if os.path.isfile(f'Trained/{filename_model}.opt'):
    op = torch.load(f'Trained/{filename_model}.opt', weights_only=False)
    optimizer.load_state_dict(op)
optimizer_to(optimizer, device)

train_model(model=model, optimizer=optimizer, train_loader=train_loader, filename=filename_model, num_epochs=num_epochs, device=device)
