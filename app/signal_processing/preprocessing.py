import numpy as np
from scipy.signal import butter, sosfiltfilt


def remove_dc_offset(samples: list[float]) -> list[float]:
    if not samples:
        return []

    average = sum(samples) / len(samples)

    return [sample - average for sample in samples]


def bandpass_filter(
    samples: list[float],
    sampling_rate: int,
    low_freq: float = 0.5,
    high_freq: float = 40.0,
    order: int = 4,
) -> list[float]:
    if not samples:
        return []

    nyquist_frequency = sampling_rate / 2

    if not 0 < low_freq < high_freq < nyquist_frequency:
        raise ValueError("Invalid band-pass frequencies.")

    signal = np.asarray(
        samples,
        dtype=float,
    )

    sos = butter(
        order,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )

    filtered_signal = sosfiltfilt(
        sos,
        signal,
    )

    return filtered_signal.tolist()
