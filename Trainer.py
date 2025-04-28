from fileinput import filename

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader
from SpeakerActivityDataset import SpeakerActivityDataset

def train(
    model, dataset, criterion, optimizer,
    filename_model, num_epochs,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):

    # num_speakers = len(dataset.speakers)
    # sample_rate = dataset.sample_rate
    model = model.to(device)
    criterion = criterion.to(device)

    # Split into train and validation
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss_list_train = []
    loss_list_validate = []

    for epoch in range(1, num_epochs + 1):
        # === Training Phase ===
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            logits = model(x)
            logits = torch.squeeze(logits, dim=1)
            # print(logits.shape, y.shape)
            loss_value = criterion(logits, y)
            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()

        avg_train_loss = train_loss / len(train_loader)
        loss_list_train.append(avg_train_loss)

        torch.save(model.state_dict(), f'{filename_model}.pth')
        torch.save(optimizer.state_dict(), f'{filename_model}.opt')

        # === Validation Phase ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).float()

                logits = model(x)
                logits = torch.squeeze(logits, dim=1)
                loss_value = criterion(logits, y)
                val_loss += loss_value.item()

        avg_val_loss = val_loss / len(val_loader)

        loss_list_validate.append(avg_val_loss)

        torch.save(model.state_dict(), f'{filename_model}.pth')
        torch.save(optimizer.state_dict(), f'{filename_model}.opt')

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        plt.plot(loss_list_train, label='train')
        plt.plot(loss_list_validate, label='validate')
        plt.legend(loc='upper right')
        plt.savefig('Loss.jpg', dpi=300)
        plt.close()

    print("Training complete.")