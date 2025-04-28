import torch
import os
from pathlib import Path



def write_audacity_labels(preds, speaker_names, time_resolution, output_path, threshold=0.5):
    """
    Write diarization predictions to Audacity-style label file.

    Args:
        preds (torch.Tensor or np.ndarray): shape [num_frames, num_speakers], values 0.0–1.0
        speaker_names (list of str): speaker names corresponding to columns in preds
        time_resolution (float): duration of each frame in seconds
        output_path (str): path to save the .txt file
        threshold (float): threshold to consider a speaker active
    """
    import numpy as np

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    preds_bin = (preds >= threshold).astype(np.uint8)  # Binarize

    preds_bin = np.squeeze(preds_bin, axis=1)

    num_frames, num_speakers = preds_bin.shape
    output = []

    for speaker_idx in range(num_speakers):
        active = preds_bin[:, speaker_idx]
        speaker = speaker_names["label"][speaker_idx]

        start_frame = None
        for i in range(num_frames):
            if active[i]:
                if start_frame is None:
                    start_frame = i
            else:
                if start_frame is not None:
                    # End of segment
                    start_time = start_frame * time_resolution
                    end_time = i * time_resolution
                    output.append((start_time, end_time, speaker))
                    start_frame = None

        # Final active segment (if ended with activity)
        if start_frame is not None:
            start_time = start_frame * time_resolution
            end_time = num_frames * time_resolution
            output.append((start_time, end_time, speaker))

    output.sort(key=lambda x: x[0])  # sort by start time
    # Write to file
    if not os.path.exists(output_path):
        with open(output_path, 'w'): pass
    with open(output_path, 'w') as f:
        for start, end, label in output:
            f.write(f"{start:.3f}\t{end:.3f}\t{label}\n")