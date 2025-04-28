import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path

class SpeakerActivityDataset(Dataset):
    def __init__(self, audio_path, class_path, label_path, sample_rate=16000, time_resolution=0.01):
        """
        :param audio_path: Path to .wav file (mono or stereo)
        :param label_path: Path to Audacity annotation .txt file
        :param time_resolution: Frame duration in seconds (e.g., 0.01 = 10ms)
        """
        self.audio_path = Path(audio_path)
        self.label_path = Path(label_path)
        self.time_resolution = time_resolution
        self.sample_rate = sample_rate

        self.class_names = pd.read_csv(class_path, sep=' ', header=None, names=['class', 'label'])

        # Load audio
        self.waveform, fs = torchaudio.load(audio_path)  # shape: [C, L]
        # Handle mono or stereo
        self.waveform = self.waveform.mean(dim=0, keepdim=True)  # force mono, shape: [1, L]
        # Resample if necessary
        if fs is not self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=self.sample_rate)
            self.audio = resampler(self.waveform)

        total_samples = self.waveform.shape[1]
        self.chunk_size = int(self.sample_rate * time_resolution)

        # Pad waveform so it's divisible by chunk size
        remainder = total_samples % self.chunk_size
        if remainder > 0:
            pad_amount = self.chunk_size - remainder
            self.waveform = torch.nn.functional.pad(self.waveform, (0, pad_amount))
        self.total_samples = self.waveform.shape[1]
        self.num_chunks = self.total_samples // self.chunk_size

        # Reshape audio into [T, C, L] = [num_chunks, 1, chunk_size]
        self.audio_chunks = self.waveform.view(1, self.num_chunks, self.chunk_size).permute(1, 0, 2)

        # Parse labels and align them to chunks
        self.label_tensor, self.speakers = self._generate_labels()

        # Ensure label and audio lengths match
        if self.label_tensor.shape[0] < self.num_chunks:
            pad = torch.zeros((self.num_chunks - self.label_tensor.shape[0], len(self.speakers)))
            self.label_tensor = torch.cat([self.label_tensor, pad], dim=0)
        elif self.label_tensor.shape[0] > self.num_chunks:
            self.label_tensor = self.label_tensor[:self.num_chunks]

    def _parse_annotations(self):
        annotations = []
        max_time = 0.0
        with open(self.label_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    start, end, label = float(parts[0]), float(parts[1]), parts[2]
                    annotations.append((start, end, label))
                    max_time = max(max_time, end)
        return annotations, max_time

    def _get_speaker_list(self, annotations):
        return sorted(set(label for _, _, label in annotations))

    def _generate_labels(self):
        annotations, _ = self._parse_annotations()
        speakers = self._get_speaker_list(annotations)
        speaker_to_idx = {name: i for i, name in enumerate(speakers)}

        T = self.num_chunks
        N = len(speakers)
        label_tensor = torch.zeros((T, N), dtype=torch.float32)

        for start, end, speaker in annotations:
            if speaker not in speaker_to_idx:
                continue
            idx = speaker_to_idx[speaker]
            start_frame = int(np.floor(start / self.time_resolution))
            end_frame = int(np.ceil(end / self.time_resolution))
            start_frame = max(0, min(start_frame, T))
            end_frame = max(0, min(end_frame, T))
            label_tensor[start_frame:end_frame, idx] = 1.0

        return label_tensor, speakers

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        audio_chunk = self.audio_chunks[idx]        # [1, chunk_size]
        label_vector = self.label_tensor[idx]       # [num_speakers]
        return audio_chunk, label_vector

    def get_num_classes(self):
        return len(self.class_names)

    def set_length(self, length):
        self.num_chunks = min(self.num_chunks, length)
