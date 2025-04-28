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

        self.rnn = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_speakers)
        )

    def forward(self, x):  # [B, 1, chunk_size]
        mfcc = self.mfcc(x)           # [B, n_mfcc, T]
        mfcc = mfcc.transpose(1, 2)   # [B, T, n_mfcc]
        rnn_out, _ = self.rnn(mfcc)   # [B, T, 2*hidden_dim]
        x = rnn_out.mean(dim=1)       # [B, 2*hidden_dim]

        return self.encoder(x)        # [B, num_speakers]