import math

import pytest
import torch

from app.ml.deep_learning.classifier import classify_eeg_signal
from app.ml.deep_learning.cnn_classifier import classify_eeg_signal_cnn
from app.ml.deep_learning.cnn_model import EEGCNNClassifier
from app.ml.deep_learning.dataset import SyntheticEEGDataset
from app.ml.deep_learning.model import EEGStateClassifier


def generate_signal(frequency: float) -> list[float]:
    return [
        math.sin(2 * math.pi * frequency * index / 256)
        for index in range(256)
    ]


def test_model_output_shape() -> None:
    model = EEGStateClassifier()

    signal = torch.randn(1, 256)

    output = model(signal)

    assert output.shape == (1, 3)


def test_synthetic_dataset() -> None:
    dataset = SyntheticEEGDataset(
        samples_per_state=10,
    )

    assert len(dataset) == 30

    signal, label = dataset[0]

    assert signal.shape == (256,)
    assert label in {0, 1, 2}


@pytest.mark.parametrize(
    ("frequency", "expected_state"),
    [
        (10.0, "relaxed"),
        (20.0, "focused"),
        (6.0, "drowsy"),
    ],
)
def test_deep_learning_classifier(
    frequency: float,
    expected_state: str,
) -> None:
    signal = generate_signal(frequency)

    result = classify_eeg_signal(signal)

    assert result["state"] == expected_state
    assert 0.0 <= result["confidence"] <= 1.0


def test_classifier_rejects_invalid_signal_length() -> None:
    with pytest.raises(
        ValueError,
        match="exactly 256 samples",
    ):
        classify_eeg_signal([0.0] * 100)


def test_cnn_model_output_shape() -> None:
    model = EEGCNNClassifier()

    signal = torch.randn(1, 256)

    output = model(signal)

    assert output.shape == (1, 3)


@pytest.mark.parametrize(
    ("frequency", "expected_state"),
    [
        (10.0, "relaxed"),
        (20.0, "focused"),
        (6.0, "drowsy"),
    ],
)
def test_cnn_classifier(
    frequency: float,
    expected_state: str,
) -> None:
    signal = generate_signal(frequency)

    result = classify_eeg_signal_cnn(signal)

    assert result["state"] == expected_state
    assert 0.0 <= result["confidence"] <= 1.0


def test_cnn_classifier_rejects_invalid_signal_length() -> None:
    with pytest.raises(
        ValueError,
        match="exactly 256 samples",
    ):
        classify_eeg_signal_cnn([0.0] * 100)
