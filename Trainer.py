
import torch
import torch.nn as nn


def train_model(model, optimizer, train_loader, filename, num_epochs=10, lr=1e-3, device='cpu'):

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            torch.save(model.state_dict(), f'Trained/{filename}.pth')
            torch.save(optimizer.state_dict(), f'Trained/{filename}.opt')

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}")
