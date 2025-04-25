import torch
from torch.utils.data import Dataset
import librosa
import pandas as pd
import numpy as np
import torchaudio


class DiarizationChunkDataset(Dataset):
    def __init__(self, audio_path, label_path, class_path, frame_size=1.0, hop_size=0.5, sample_rate=16000):
        """
        Args:
            audio_path (str): Path to the .mp3 audio file.
            label_path (str): Path to the Audacity .txt label file.
            frame_size (float): Frame size in seconds (e.g., 1.0 = 1 second).
            hop_size (float): Hop size in seconds (e.g., 0.5 = 50% overlap).
            sample_rate (int): Target sample rate for audio.
        """
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

        # Load audio
        self.audio, fs = torchaudio.load(audio_path)
        # Make mono
        self.audio = torch.mean(self.audio, dim=0)
        # Resample if necessary
        if fs is not self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=self.sample_rate)
            self.audio = resampler(self.audio)

        # Load labels
        self.labels = pd.read_csv(label_path, sep='\t', header=None, names=['start', 'end', 'label'])

        # Load class names
        self.class_names = pd.read_csv(class_path, sep=' ', header=None, names=['class', 'label'])

        # Map speakers to integer labels
        self.speaker_map = {name: i for i, name in enumerate(self.labels['label'].unique())}

        # Build chunk index
        self.frames = self._create_frames()


    def _create_frames(self):
        """Split audio into overlapping frames with associated speaker label."""
        # total_duration = len(self.audio) / self.sample_rate
        frame_samples = int(self.frame_size * self.sample_rate)
        hop_samples = int(self.hop_size * self.sample_rate)

        frames = []

        for start_sample in range(0, len(self.audio) - frame_samples + 1, hop_samples):
            start_time = start_sample / self.sample_rate
            end_time = (start_sample + frame_samples) / self.sample_rate

            # Find overlapping label
            label_row = self.labels[
                (self.labels['start'] <= start_time) &
                (self.labels['end'] >= end_time)
                ]

            # Assign speaker label if fully inside a labeled region
            if not label_row.empty:
                speaker = label_row.iloc[0]['label']
                speaker_idx = self.speaker_map[speaker]
                frames.append((start_sample, start_sample + frame_samples, speaker_idx))

        return frames

    def __len__(self):
        return len(self.frames)

    def get_num_classes(self):
        return len(self.class_names)

    def __getitem__(self, idx):
        start, end, label = self.frames[idx]
        chunk = self.audio[start:end]
        label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), self.get_num_classes())
        return chunk, label_one_hot