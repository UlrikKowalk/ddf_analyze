import os.path
import torch
from numba.cuda.models import Dim3Model
from torch.utils.data import DataLoader
from SpeakerDiarizationDNN import SpeakerDiarizationDNN
from Trainer import train
from MFCCDiarizationModel_old import MFCCDiarizationModel
from DiarizationChunkDataset import DiarizationChunkDataset
from compute_class_frequencies import compute_class_frequencies


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
class_name_file = "classes.txt"

filename_model = "Trained/ddf_diarizer_new"

device = "cuda"
learning_rate = 0.0001
num_epochs = 100

dataset = DiarizationChunkDataset(
    audio_path=audio_file,
    label_path=label_file,
    class_path=class_name_file,
    sample_rate=16000,
    frame_size=0.5,
    hop_size=0.5
)

if not os.path.exists('Trained'):
    os.mkdir('Trained')

model = MFCCDiarizationModel(sample_rate=16000, n_mfcc=40, num_speakers=dataset.get_num_classes(), hidden_dim=64)
if os.path.isfile(f'{filename_model}.pth'):
    sd = torch.load(f'{filename_model}.pth', weights_only=False)
    model.load_state_dict(sd)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if os.path.isfile(f'{filename_model}.opt'):
    op = torch.load(f'{filename_model}.opt', weights_only=False)
    optimizer.load_state_dict(op)
optimizer_to(optimizer, device)

# speaker_class_frequencies = compute_class_frequencies(dataset=dataset)
# speaker_class_frequencies[0] = 1000
# pos_weight = torch.tensor([100.0 / class_freq for class_freq in speaker_class_frequencies])
criterion = torch.nn.BCEWithLogitsLoss()

train(model=model,
      dataset=dataset,
      criterion=criterion,
      optimizer=optimizer,
      filename_model=filename_model,
      num_epochs=num_epochs,
      device=device)
