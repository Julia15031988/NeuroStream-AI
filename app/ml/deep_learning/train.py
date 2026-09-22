from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from app.ml.deep_learning.dataset import SyntheticEEGDataset
from app.ml.deep_learning.model import EEGStateClassifier

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "eeg_state_classifier.pt"
)


def train_model() -> None:
    dataset = SyntheticEEGDataset()

    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
    )

    model = EEGStateClassifier()

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 10

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for signals, labels in train_loader:
            optimizer.zero_grad()

            predictions = model(signals)

            loss = loss_function(
                predictions,
                labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for signals, labels in validation_loader:
                predictions = model(signals)

                predicted_labels = predictions.argmax(dim=1)

                correct += (
                    predicted_labels == labels
                ).sum().item()

                total += labels.size(0)

        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- loss: {average_loss:.4f} "
            f"- validation accuracy: {accuracy:.4f}"
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
