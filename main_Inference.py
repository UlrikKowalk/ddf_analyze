import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from DiarizationChunkDataset import DiarizationChunkDataset
from MFCCDiarizationModel_old import MFCCDiarizationModel
from write_audacity_labels import write_audacity_labels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_model(model_path, sample_rate, n_mfcc, num_speakers, device):
    model = MFCCDiarizationModel(sample_rate=sample_rate,
                                 n_mfcc=n_mfcc,
                                 num_speakers=num_speakers,
                                 hidden_dim=64
                                 ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def run_inference(model, dataset, threshold=0.5, device="cpu"):
    preds = []
    with torch.no_grad():
        for x, _ in dataset:
            x = x.unsqueeze(0).to(device)  # [1, 1, chunk_size]
            logits = model(x)              # [1, num_speakers]
            pred = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            preds.append(pred)
    preds = np.stack(preds)  # [T, num_speakers]
    binary_preds = (preds >= threshold).astype(int)
    return binary_preds, preds


def plot_diarization(preds, speakers, time_resolution):
    num_speakers = len(speakers)
    time_axis = np.arange(preds.shape[0]) * time_resolution

    fig, axes = plt.subplots(num_speakers, 1, figsize=(12, 1.5 * num_speakers), sharex=True)
    speakers = speakers["label"]
    for i, speaker in enumerate(speakers):

        ax = axes[i] if num_speakers > 1 else axes

        ax.step(time_axis, preds[:, 0, i], where='post', label='Prediction', linestyle='-', alpha=0.7)
        ax.set_ylabel(speaker)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f'quick.jpg', dpi=300)
    plt.close()


def test_diarization(
    model_path,
    audio_path,
    label_path,
    class_path,
    inference=True,
    time_resolution=0.5,
    threshold=0.5,
    n_mfcc=20,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print("Preparing dataset...")
    dataset = DiarizationChunkDataset(audio_path=audio_path, label_path=label_path, class_path=class_path, frame_size=time_resolution)
    speakers = dataset.get_classes()
    sample_rate = dataset.sample_rate
    print(f"Loaded audio with {len(dataset)} chunks and {dataset.get_num_classes()} speakers.")

    print("Loading model...")
    model = load_model(model_path, sample_rate, n_mfcc, dataset.get_num_classes(), device)

    print("Running inference...")
    pred_binary, _ = run_inference(model, dataset, threshold, device)

    print("Plotting results...")
    plot_diarization(pred_binary, speakers, time_resolution)

    write_audacity_labels(preds=pred_binary, speaker_names=dataset.get_classes(), time_resolution=time_resolution, output_path=label_path, threshold=threshold)


if __name__ == "__main__":
    test_diarization(
        model_path="Trained/ddf_diarizer_new.pth",
        audio_path="010 - Die drei Fragezeichen und die fluesternde Mumie (B).mp3",
        label_path="010 - Die drei Fragezeichen und die fluesternde Mumie (B).txt",
        class_path="classes.txt",
        time_resolution=0.5,
        threshold=0.7,
        n_mfcc=40
    )