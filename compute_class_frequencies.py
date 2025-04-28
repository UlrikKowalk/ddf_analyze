import torch
from torch.utils.data import DataLoader
from collections import Counter

def compute_class_frequencies(dataset):
    num_classes = dataset.get_num_classes()
    counts = torch.zeros(num_classes)

    for i in range(len(dataset)):
        _, label = dataset[i]  # label is one-hot (e.g., [0, 1, 0, ...])
        counts += label.float()

    return counts