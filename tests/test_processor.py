import math

from app.signal_processing.processor import process_eeg_samples


def generate_sine_wave(
    frequency: float,
    sampling_rate: int = 256,
    duration: float = 1.0,
) -> list[float]:
    samples_count = int(
        sampling_rate * duration
    )

    return [
        math.sin(
            2 * math.pi * frequency * i / sampling_rate
        )
        for i in range(samples_count)
    ]


def test_alpha_signal_is_dominant() -> None:
    samples = generate_sine_wave(
        frequency=10.0,
    )

    result = process_eeg_samples(
        samples=samples,
        sampling_rate=256,
    )

    assert result["samples_count"] == 256
    assert result["dominant_wave"] == "alpha"
    assert result["alpha_power"] > result["beta_power"]
    assert result["alpha_power"] > result["theta_power"]
    assert result["alpha_power"] > result["delta_power"]

def test_beta_signal_is_dominant() -> None:
    samples = generate_sine_wave(
        frequency=20.0,
    )

    result = process_eeg_samples(
        samples=samples,
        sampling_rate=256,
    )

    assert result["samples_count"] == 256
    assert result["dominant_wave"] == "beta"
    assert result["beta_power"] > result["alpha_power"]
    assert result["beta_power"] > result["theta_power"]
    assert result["beta_power"] > result["delta_power"]

def test_theta_signal_is_dominant() -> None:
    samples = generate_sine_wave(
        frequency=6.0,
    )

    result = process_eeg_samples(
        samples=samples,
        sampling_rate=256,
    )

    assert result["samples_count"] == 256
    assert result["dominant_wave"] == "theta"
    assert result["theta_power"] > result["alpha_power"]
    assert result["theta_power"] > result["beta_power"]
    assert result["theta_power"] > result["delta_power"]

def test_delta_signal_is_dominant() -> None:
    samples = generate_sine_wave(
        frequency=2.0,
    )

    result = process_eeg_samples(
        samples=samples,
        sampling_rate=256,
    )

    assert result["samples_count"] == 256
    assert result["dominant_wave"] == "delta"
    assert result["delta_power"] > result["theta_power"]
    assert result["delta_power"] > result["alpha_power"]
    assert result["delta_power"] > result["beta_power"]

def test_empty_samples() -> None:
    result = process_eeg_samples(
        samples=[],
        sampling_rate=256,
    )

    assert result["samples_count"] == 0
    assert result["average"] == 0.0
    assert result["minimum"] == 0.0
    assert result["maximum"] == 0.0
    assert result["std"] == 0.0
    assert result["rms"] == 0.0
    assert result["dominant_wave"] is None
    assert result["predicted_state"] is None
    assert result["prediction_confidence"] == 0.0
