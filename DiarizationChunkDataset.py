import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class DiarizationChunkDataset(Dataset):
    def __init__(self, audio_path, label_path, class_path, frame_size=1.0, hop_size=0.5, sample_rate=16000):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

        # Load audio
        self.audio, fs = torchaudio.load(audio_path)
        self.audio = torch.mean(self.audio, dim=0)  # mono

        if fs != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=self.sample_rate)
            self.audio = resampler(self.audio)

        # Load labels
        self.labels = pd.read_csv(label_path, sep='\t', header=None, names=['start', 'end', 'label'])

        # Load speaker class mapping
        self.class_names = pd.read_csv(class_path, sep=' ', header=None, names=['class', 'label'])
        self.speaker_map = {name: i for i, name in enumerate(self.class_names['class'])}

        # Build frame index
        self.frames = self._create_frames()
        self.length = len(self.frames)

        idx = 0
        self.label_tensor = torch.zeros(size=(self.length, self.get_num_classes()))
        for _, _, label in self.frames:
            self.label_tensor[idx, :] = label
            idx+=1

    def _create_frames(self):
        frame_samples = int(self.frame_size * self.sample_rate)
        hop_samples = int(self.hop_size * self.sample_rate)

        frames = []
        for start_sample in range(0, len(self.audio) - frame_samples + 1, hop_samples):
            start_time = start_sample / self.sample_rate
            end_time = (start_sample + frame_samples) / self.sample_rate

            label_rows = self.labels[
                (self.labels['end'] > start_time) &
                (self.labels['start'] < end_time)
            ]

            label_vector = torch.zeros(self.get_num_classes())
            for _, row in label_rows.iterrows():
                speaker = row['label']
                if speaker in self.speaker_map:
                    label_vector[self.speaker_map[speaker]] = 1.0

            frames.append((start_sample, start_sample + frame_samples, label_vector))
        return frames

    def __len__(self):
        return self.length

    def get_num_classes(self):
        return len(self.class_names)

    def get_classes(self):
        return self.class_names

    def __getitem__(self, idx):
        start, end, label_vector = self.frames[idx]
        chunk = self.audio[start:end].unsqueeze(0)  # [1, chunk_size]
        return chunk, label_vector