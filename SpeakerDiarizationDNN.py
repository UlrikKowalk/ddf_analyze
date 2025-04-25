import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SpeakerDiarizationDNN(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=40, num_classes=14):
        super(SpeakerDiarizationDNN, self).__init__()

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.num_classes = num_classes

        # Feature extractor (MFCC)
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
        )

        # Compute input dimension dynamically after MFCC
        # (We'll assume a fixed-length input or pad/crop as needed)
        # Let's say input is 1 second (sample_rate samples) → T = ~100 frames
        self.input_dim = self.n_mfcc  # for each frame

        # DNN layers
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.num_classes)

    def extract_features(self, audio):
        """
        Expects raw audio waveform of shape [N]
        Returns MFCC features of shape [T, n_mfcc]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # add batch dimension
        mfcc = self.mfcc_extractor(audio)  # [1, n_mfcc, T]
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # [T, n_mfcc]
        return mfcc

    def forward(self, audio):
        """
        audio: raw audio waveform [N] or batch of waveforms [B, N]
        returns: logits [T, num_speakers] or [B, T, num_speakers]
        """
        single_input = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            single_input = True

        features = []
        for waveform in audio:
            mfcc = self.extract_features(waveform)  # [T, n_mfcc]
            features.append(mfcc)

        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)  # [B, T, n_mfcc]

        B, T, D = features.shape
        features = features.view(B * T, D)

        x = F.relu(self.fc1(features))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc3(x)  # [B*T, num_speakers]

        x = x.view(B, T, self.num_classes)
        return x[0] if single_input else x