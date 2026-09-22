import math

import pytest

from app.signal_processing.preprocessing import (
    bandpass_filter,
    remove_dc_offset,
)


def test_remove_dc_offset() -> None:
    samples = [1.0, 2.0, 3.0]

    result = remove_dc_offset(samples)

    assert result == [-1.0, 0.0, 1.0]


def test_remove_dc_offset_with_empty_samples() -> None:
    result = remove_dc_offset([])

    assert result == []

def test_bandpass_filter_returns_filtered_signal() -> None:
    sampling_rate = 256

    samples = [
        math.sin(
            2 * math.pi * 10 * i / sampling_rate
        )
        for i in range(256)
    ]

    result = bandpass_filter(
        samples=samples,
        sampling_rate=sampling_rate,
    )

    assert len(result) == len(samples)
    assert all(
        isinstance(sample, float)
        for sample in result
    )

def test_bandpass_filter_with_invalid_frequencies() -> None:
    samples = [0.1] * 256

    with pytest.raises(
        ValueError,
        match="Invalid band-pass frequencies.",
    ):
        bandpass_filter(
            samples=samples,
            sampling_rate=256,
            low_freq=40.0,
            high_freq=0.5,
        )
