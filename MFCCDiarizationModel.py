import torch
import torch.nn as nn
import torchaudio

class MFCCDiarizationModel(nn.Module):
    def __init__(self, sample_rate, n_mfcc=40, num_speakers=14, hidden_dim=128):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 400,
                'hop_length': 160,
                'n_mels': 64
            }
        )

        self.encoder = nn.Sequential(
            nn.Linear(n_mfcc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_speakers)
        )

    def forward(self, x):  # x: [B, 1, chunk_size]
        # print(x.shape)
        # Apply MFCC per chunk
        B = x.size(0)
        mfcc = self.mfcc(x)  # [B, n_mfcc, T]
        # print(mfcc.shape)
        # mfcc = mfcc[:, 1:, :]  # remove 0th coef → shape now [B, 40, T]
        mfcc = mfcc.mean(dim=3)  # average over time

        logits = self.encoder(mfcc)  # [B, num_speakers]
        return logits