import torchaudio

from AutolabelDataset import AutolabelDataset
import os.path
import torch
from torch.utils.data import DataLoader
from SpeakerDiarizationDNN import SpeakerDiarizationDNN
import matplotlib.pyplot as plt

audio_file = "010 - Die drei Fragezeichen und die fluesternde Mumie (A).mp3"
label_file = "010 - Die drei Fragezeichen und die fluesternde Mumie (B).txt"
class_names = "classes.txt"

filename_model = "ddf_diarizer"

device = "cuda"
length = 2000

dataset = AutolabelDataset(
    audio_path=audio_file,
    label_path=label_file,
    class_path=class_names,
    frame_size=0.1,
    hop_size=0.05,
    sample_rate=16000
)
dataset.set_length(length)

test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

audio, fs = torchaudio.load(audio_file)
audio = torch.mean(audio, dim=0)
print(audio.shape)

model = SpeakerDiarizationDNN(num_classes=dataset.get_num_classes())
if os.path.isfile(f'Trained/{filename_model}.pth'):
    sd = torch.load(f'Trained/{filename_model}.pth', weights_only=False)
    model.load_state_dict(sd)
model.to(device)

predictions = torch.zeros(size=(len(dataset), dataset.get_num_classes()))

for idx, chunk in enumerate(test_loader):
    chunk = chunk.to(device)
    prediction = model.forward(chunk)
    prediction = torch.mean(prediction[0], dim=0)
    predictions[idx, :] = prediction

    print(f'{idx}/{len(dataset)}')

fig, ax = plt.subplots(2, 1)
ax[0].plot(audio[:int(((length-1)*0.05+0.1)*fs)].cpu().detach().numpy())
ax[0].set_xlim([0, int(((length-1)*0.05+0.1)*fs)])
ax[1].imshow(torch.transpose(predictions, dim0=0, dim1=1).cpu().detach().numpy(), interpolation='nearest', aspect='auto')
fig.set_size_inches(10, 3)
plt.show()