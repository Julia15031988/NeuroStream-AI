from pathlib import Path

import torch

from app.ml.deep_learning.cnn_model import EEGCNNClassifier

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "eeg_cnn_classifier.pt"
)

LABEL_TO_STATE = {
    0: "relaxed",
    1: "focused",
    2: "drowsy",
}


model = EEGCNNClassifier()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
    )
)

model.eval()


def classify_eeg_signal_cnn(
    samples: list[float],
) -> dict:
    if len(samples) != 256:
        raise ValueError(
            "CNN classifier expects exactly 256 samples."
        )

    signal = torch.tensor(
        samples,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(signal)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_label = int(
            probabilities.argmax(dim=1).item()
        )

        confidence = float(
            probabilities[0][predicted_label].item()
        )

    return {
        "state": LABEL_TO_STATE[predicted_label],
        "confidence": round(confidence, 4),
    }
