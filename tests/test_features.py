import math

import numpy as np

from app.signal_processing.features import (
    calculate_band_power,
    calculate_psd,
)


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


def test_calculate_psd_returns_data() -> None:
    samples = generate_sine_wave(
        frequency=10.0,
    )

    frequencies, power_spectral_density = calculate_psd(
        samples=samples,
        sampling_rate=256,
    )

    assert len(frequencies) > 0
    assert len(power_spectral_density) > 0
    assert len(frequencies) == len(power_spectral_density)


def test_alpha_band_power_is_positive() -> None:
    samples = generate_sine_wave(
        frequency=10.0,
    )

    frequencies, power_spectral_density = calculate_psd(
        samples=samples,
        sampling_rate=256,
    )

    alpha_power = calculate_band_power(
        frequencies=frequencies,
        power_spectral_density=power_spectral_density,
        low_freq=8.0,
        high_freq=13.0,
    )

    assert alpha_power > 0

def test_calculate_psd_with_empty_samples() -> None:
    frequencies, power_spectral_density = calculate_psd(
        samples=[],
        sampling_rate=256,
    )

    assert len(frequencies) == 0
    assert len(power_spectral_density) == 0

def test_calculate_band_power_with_empty_arrays() -> None:
    frequencies = np.array([])
    power_spectral_density = np.array([])

    band_power = calculate_band_power(
        frequencies=frequencies,
        power_spectral_density=power_spectral_density,
        low_freq=8.0,
        high_freq=13.0,
    )

    assert band_power == 0.0
